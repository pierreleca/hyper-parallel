# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Build a layer-cropped Qwen3-VL-MoE model from its full checkpoint.

The full model needs eight 64 GB dies: weights, gradients and BF16 AdamW
moments come to 8 bytes per parameter, so 30.5 B parameters are about 30.5 GB
per die over eight ranks and about 61 GB over four. Cropping the text decoder
is what makes a four-device run possible, and the layer count lives in the
nested ``text_config``, which the YAML target mechanism cannot reach: it passes
nested mappings through as plain values rather than descending into them.

The builder mirrors ``examples/training_demo/cropped_qwen3_moe.py``, the
established convention for the same problem on Qwen3-MoE, but loads the
retained layers from the checkpoint instead of initializing them at random.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PreTrainedModel

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models import HyperAutoModelForImageTextToText
from hyper_parallel.models.build_options import CompileConfig

_EXPECTED_MODEL_TYPE = "qwen3_vl_moe"


def build_cropped_qwen3_vl(
        pretrained_model_name_or_path: str,
        num_hidden_layers: int = 4,
        local_files_only: bool = True,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        validate_placement: bool = False,
        allow_uncovered_params: bool = True,
        distributed_setup: DistributedSetup | None = None,
        peft_config: Any | None = None,
        compile_config: CompileConfig | dict[str, Any] | None = None,
        activation_checkpoint: str | None = None,
        activation_swap: str = "none",
) -> PreTrainedModel:
    """Create a Qwen3-VL-MoE model with fewer text layers, from its checkpoint.

    The model is built by ``from_pretrained`` with the cropped configuration:
    the retained text layers, the vision tower, the embeddings and the LM head
    come from the checkpoint, and the tensors of the dropped layers are
    skipped. Real weights matter wherever routing does: random router weights
    spread tokens almost evenly over the experts, which a trained router does
    not.

    The VLM Trainer reads the processor location from
    ``model.tokenizer_path``, then ``model.pretrained_model_name_or_path``, so
    one YAML key serves as the configuration, processor and checkpoint source.

    Args:
        pretrained_model_name_or_path: Local Hugging Face Qwen3-VL-MoE model
            directory.
        num_hidden_layers: Text decoder layers retained in the cropped model,
            counted from the first.
        local_files_only: Disable implicit Hub downloads when true.
        torch_dtype: Model parameter dtype accepted by HyperAutoModel.
        attn_implementation: Hugging Face attention implementation name.
        validate_placement: Enable HyperParallel placement validation.
        allow_uncovered_params: Keep vision-tower parameters that no sharding
            spec declares as plain FSDP parameters, and downgrade the planner
            coverage check to a warning. Under EP the planner covers the text
            decoder but, in the vision tower, only the LayerNorms.
        distributed_setup: Trainer-provided distributed topology.
        peft_config: Optional Trainer-provided PEFT configuration.
        compile_config: Optional Trainer-provided compile configuration.
        activation_checkpoint: Activation checkpoint mode.
        activation_swap: Activation swap mode.

    Returns:
        A parallelized, cropped Qwen3-VL-MoE model with checkpoint weights.

    Raises:
        ValueError: If the layer count is not positive or exceeds the
            checkpoint's, the directory holds a different architecture, or
            the configuration has no text_config.
    """
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    model_type = getattr(config, "model_type", None)
    if model_type != _EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"pretrained_model_name_or_path must contain a {_EXPECTED_MODEL_TYPE} configuration; "
            f"got model_type={model_type!r}"
        )
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise ValueError(
            "Qwen3-VL configuration exposes no text_config to crop"
        )

    if num_hidden_layers > text_config.num_hidden_layers:
        raise ValueError(
            f"num_hidden_layers={num_hidden_layers} exceeds the "
            f"{text_config.num_hidden_layers} text layers of the checkpoint"
        )

    text_config.num_hidden_layers = num_hidden_layers
    # Training reads no cache, and the flag lives on both levels of this
    # configuration.
    config.use_cache = False
    text_config.use_cache = False

    return HyperAutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path,
        config=config,
        local_files_only=local_files_only,
        distributed_setup=distributed_setup,
        peft_config=peft_config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        validate_placement=validate_placement,
        allow_uncovered_params=allow_uncovered_params,
        compile_config=compile_config,
        activation_checkpoint=activation_checkpoint,
        activation_swap=activation_swap,
    )


__all__ = ["build_cropped_qwen3_vl"]
