# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VLM Trainer assembled from the shared BaseTrainer stages."""

__all__ = ["VLMTrainer"]

from collections import defaultdict
from typing import Any, Dict

import torch

from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.data.batching import calculate_num_micro_batches
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.vlm import build_processor, build_vlm_get_batch
from hyper_parallel.trainer.runtime.distributed import all_reduce
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.logging import create_logger
from hyper_parallel.trainer.runtime.memory import print_device_mem_info
from hyper_parallel.trainer.runtime.device import synchronize  # pylint: disable=syntax-error
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import TrainerConfig

logger = create_logger(__name__)


class VLMTrainer:
    """Compose the VLM training runtime from explicit BaseTrainer stages."""

    base: BaseTrainer

    def __init__(self, config: TrainerConfig) -> None:
        """Build the VLM Trainer in the same explicit order as VeOmni."""
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.config = config

        self.base._setup()
        self.base._build_model()
        self.base._build_loss()

        # datasets
        self._build_model_assets()
        self._build_data_transform()
        self.base._build_dataset()

        # dataloader
        self._build_collate_fn()
        self.base._build_dataloader()

        # get_batch
        self._build_get_batch()
        self.base._compute_train_iters()

        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()

    def _build_model_assets(self) -> None:
        """Build processor-backed assets for VLM training."""
        config: TrainerConfig = self.base.config
        if config.dataset is None:
            raise ValueError("dataset must define a build target")

        processor_path = (
            getattr(config.model, "tokenizer_path", None)
            or getattr(config.model, "pretrained_model_name_or_path", None)
        )
        if processor_path is None:
            self.base.processor = None
        else:
            self.base.processor = build_processor(processor_path)
        self.base.tokenizer = getattr(self.base.processor, "tokenizer", None)
        self.base.chat_template = None

        self.base.model_assets = [self.base.model_config]
        if self.base.processor is not None:
            self.base.model_assets.append(self.base.processor)

    def _build_data_transform(self) -> None:
        """Build the configured multimodal sample transform."""
        dataset_config = self.base.config.dataset
        if dataset_config is None:
            raise ValueError("dataset must define a build target")
        if dataset_config.data_transform is None:
            self.base.data_transform = None
            return
        self.base.data_transform = dataset_config.data_transform.build(
            processor=self.base.processor,
        )

    def _build_collate_fn(self) -> None:
        """Build the VLM collator and gradient-accumulation batch count."""
        dataloader_config = self.base.config.dataloader
        if dataloader_config is None or dataloader_config.collate_fn is None:
            raise ValueError("dataloader.collate_fn must define a build target")
        training_config = self.base.config.training
        self.base.num_micro_batches = calculate_num_micro_batches(
            global_batch_size=training_config.global_batch_size,
            micro_batch_size=training_config.micro_batch_size,
            dp_world_size=self.base.mesh.dp_size,
        )
        self.base.collate_fn = dataloader_config.collate_fn.build()

    def _build_get_batch(self) -> None:
        """Build the DataLoader-to-VLM batch adapter."""
        config = self.base.config
        get_batch_builder = (
            config.dataloader.get_batch.build
            if config.dataloader.get_batch
            else build_vlm_get_batch
        )
        self.base.get_batch = get_batch_builder(
            mesh_context=self.base.mesh,
            device=self.base.device,
            pp_shared_data=bool(getattr(config.dataloader, "pp_shared_data", False)),
        )

    @property
    def distributed_setup(self) -> Any:
        """Return the shared distributed setup."""
        return self.base.distributed_setup

    @property
    def mesh(self) -> Any:
        """Return the shared mesh context."""
        return self.base.mesh

    @property
    def dp_cp_mesh(self) -> Any:
        """Return the shared data/context-parallel mesh."""
        return self.base.dp_cp_mesh

    def on_train_begin(self) -> None:
        """Dispatch the training-begin lifecycle hook."""
        self.base.on_train_begin()

    def on_train_end(self) -> None:
        """Dispatch the training-end lifecycle hook."""
        self.base.on_train_end()

    def on_epoch_begin(self) -> None:
        """Dispatch the epoch-begin lifecycle hook."""
        self.base.on_epoch_begin()

    def on_epoch_end(self) -> None:
        """Dispatch the epoch-end lifecycle hook."""
        self.base.on_epoch_end()

    def on_step_begin(self, micro_batches: Any = None) -> None:
        """Dispatch the step-begin lifecycle hook."""
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(
            self,
            loss: Any = None,
            loss_dict: Any = None,
            grad_norm: Any = None,
    ) -> None:
        """Dispatch the step-end lifecycle hook."""
        self.base.on_step_end(
            loss=loss,
            loss_dict=loss_dict,
            grad_norm=grad_norm,
        )

    def _forward_backward_micro_batches(
            self,
            training_batches: list[Any],
            num_micro_steps: int,
    ) -> tuple[float, Dict[str, float]]:
        """Run and aggregate the VLM forward-backward micro-steps."""
        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        for micro_step, batch in enumerate(training_batches):
            model_inputs, loss_inputs = batch
            self.base.model_reshard(micro_step, num_micro_steps)
            self.base.configure_fsdp_gradient_sync(
                micro_step,
                num_micro_steps,
            )
            self.base.current_token_counts = count_loss_token(loss_inputs)
            self.base.step_token_counts = {
                name: token_count * num_micro_steps
                for name, token_count in self.base.current_token_counts.items()
            }
            loss, loss_dict = self.base.forward_backward_step(model_inputs)

            # Release each device batch as soon as its backward pass completes;
            # prefetching all micro-batches must not pin them until step end.
            training_batches[micro_step] = None

            total_loss += loss.item()
            for loss_name, loss_value in loss_dict.items():
                total_loss_dict[loss_name] += loss_value.item()

        return total_loss, total_loss_dict

    def prefetch_micro_batches(self, data_iterator: Any) -> list:
        """Read every micro-batch of one step before any of them runs.

        Nothing here is collective, so a rank whose data has run out raises :exc:`StopIteration`
        without having left its peers waiting. That makes this the point at which the ranks can agree
        to stop; see :meth:`data_exhausted_anywhere`.

        Args:
            data_iterator: The training data iterator.

        Returns:
            One ``(model_inputs, loss_inputs)`` pair per micro-batch of the step.
        """
        return [self.base.get_batch(data_iterator) for _ in range(self.base.num_micro_batches)]

    def data_exhausted_everywhere(self, exhausted: bool) -> bool:
        """Return whether every data-parallel rank has run out, which is when the epoch ends.

        A rank reaching the end of its data is a local event, but leaving the step loop is not: the
        step that follows is full of collectives, and a rank that has left will not join them. With a
        fixed batch size every rank runs out on the same step and the question never arises. With a
        token budget it does, because an equal number of samples packs into an unequal number of
        rows: a rank holding long samples fills more rows from the same samples than one holding
        short ones.

        Stopping at the first rank to finish would therefore throw away the data the others still
        hold -- on this study's spread, nearly a third of the epoch. So the epoch runs until the last
        rank is done and the ranks that finished early keep joining the collectives with the padded
        work of :meth:`padding_micro_batches`.

        Args:
            exhausted: Whether this rank's data iterator is finished.

        Returns:
            True when every rank of the data-parallel group is finished.
        """
        group = self.base.mesh.dp_cp_mesh.get_group() if self.base.mesh.dp_cp_mesh is not None else None
        if group is None:
            return exhausted
        return bool(all_reduce(1.0 if exhausted else 0.0, op="min", group=group))

    def padding_micro_batches(self, template: Any) -> list:
        """Return micro-batches that join every collective and move the weights by nothing.

        A rank whose data has run out still has to enter the step, because the ranks that still have
        data will not get through their collectives without it. It replays its last micro-batch with
        **every** label masked, which makes the work real and the gradient exactly nothing: the
        cross-entropy writes a gradient only at the positions it supervises, and this batch
        supervises none, so no replayed sample reaches the weights twice.

        Nothing here relies on the loss weighting being small, because there is nothing to weigh.
        ``count_loss_token`` counts no supervised token, so the batch contributes nothing to the
        step's denominator either, and the step's reported loss stays the token-weighted mean over
        the ranks that did have data.

        The forward does produce NaN -- a mean over no supervised token -- and that NaN never leaves
        the forward: :class:`~hyper_parallel.components.losses.model_output.ModelOutputLoss` replaces
        the loss of a batch with no valid label by zero, and the gradient was already zero rather
        than NaN. ``check_padded_work.py`` asserts both on the host.

        Args:
            template: The last micro-batches this rank read.

        Returns:
            Micro-batches of the same shapes, supervising nothing.

        Raises:
            ValueError: If this rank never read a micro-batch, which means it was handed no data.
        """
        if not template:
            raise ValueError(
                "a rank ran out of data before its first step while others had some; the sampler "
                "handed it nothing, which is a configuration problem rather than a ragged epoch"
            )
        padded = []
        for model_inputs, loss_inputs in template:
            labels = loss_inputs.get("labels")
            if not isinstance(labels, torch.Tensor):
                padded.append((model_inputs, loss_inputs))
                continue
            masked = torch.full_like(labels, IGNORE_INDEX)
            model_padded = {**model_inputs}
            if isinstance(model_padded.get("labels"), torch.Tensor):
                model_padded["labels"] = masked
            loss_padded = {**loss_inputs, "labels": masked}
            if isinstance(loss_padded.get("loss_mask"), torch.Tensor):
                loss_padded["loss_mask"] = masked >= 0
            padded.append((model_padded, loss_padded))
        return padded

    def train_step(self, data_iterator: Any, training_batches: Any = None) -> Dict[str, float]:
        """Execute one VLM training step.

        Args:
            data_iterator: The training data iterator, read when ``training_batches`` is not given.
            training_batches: Micro-batches already read by :meth:`prefetch_micro_batches`.

        Returns:
            The step's metrics.
        """
        config = self.base.config
        if training_batches is None:
            training_batches = self.prefetch_micro_batches(data_iterator)
        num_micro_steps = self.base.num_micro_batches

        self.on_step_begin(
            micro_batches=[model_inputs for model_inputs, _ in training_batches]
        )
        synchronize()

        total_loss, total_loss_dict = self._forward_backward_micro_batches(
            training_batches,
            num_micro_steps,
        )

        if config.debug.check_nan_inf:
            self.base.report_non_finite(total_loss)

        grad_norm = clip_grad_norm_(
            self.base.model,
            config.training.max_grad_norm,
        )

        self.base.step_optimizers_and_schedulers()

        grad_norm_value = float(grad_norm)
        self.on_step_end(
            loss=total_loss,
            loss_dict=total_loss_dict,
            grad_norm=grad_norm_value,
        )

        self.base.state.global_step += 1

        return {
            "loss": total_loss,
            "grad_norm": grad_norm_value,
        }

    def train(self) -> None:
        """Run the VLM training loop."""
        config = self.base.config
        self.on_train_begin()
        logger.info(
            "Rank%s Start training. Global step: %s. Train iters: %s. Start epoch: %s. Train epochs: %s.",
            self.base.local_rank,
            self.base.state.global_step,
            self.base.train_iters,
            self.base.state.epoch,
            self.base.train_epochs,
        )

        # Checkpoint resume restores state.global_step, state.epoch, and the DataLoader cursor.
        for epoch in range(self.base.state.epoch, self.base.train_epochs):
            train_dataloader = self.base.train_dataloader

            if hasattr(train_dataloader, "set_epoch"):
                train_dataloader.set_epoch(epoch)

            self.on_epoch_begin()
            data_iterator = iter(train_dataloader) if train_dataloader is not None else None

            start_step = self.base.state.global_step - epoch * self.base.train_steps
            train_steps = min(self.base.train_steps, self.base.train_iters - epoch * self.base.train_steps)
            template = None
            for _ in range(start_step, train_steps):
                # Read the step's micro-batches first: that is collective-free, so the ranks can
                # agree about the epoch before any of them enters a step the others have left.
                try:
                    training_batches = self.prefetch_micro_batches(data_iterator)
                    exhausted = False
                except StopIteration:
                    training_batches, exhausted = None, True
                if self.data_exhausted_everywhere(exhausted):
                    logger.info("epoch:%s Dataloader finished with drop_last %s", epoch, config.dataloader.drop_last)
                    break
                if exhausted:
                    training_batches = self.padding_micro_batches(template)
                else:
                    template = training_batches
                self.train_step(data_iterator, training_batches=training_batches)

            self.on_epoch_end()
            self.base.state.epoch = epoch + 1
            print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

            if self.base.state.global_step >= self.base.train_iters:
                break

        self.on_train_end()

        synchronize()
        self.base.destroy_distributed()
