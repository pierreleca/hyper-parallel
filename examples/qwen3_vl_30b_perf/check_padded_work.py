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
"""Check that a rank's padded work moves no weight, on a small model, on the host.

A token budget makes the ranks finish an epoch several steps apart, so a rank that has run out
replays its last micro-batch to keep joining the collectives. That replay must not train on the
sample a second time, and the claim is exact rather than small: with every label masked, the
cross-entropy writes a gradient only where it supervises a token, so a batch that supervises none
leaves every parameter gradient at zero.

Four things are checked, on the real model, and two of them are controls so the check cannot pass
by doing nothing:

    1. the real batch moves the weights         -- some gradient is non-zero
    2. the padded batch moves nothing           -- EVERY gradient is bit-exact zero
    3. the loss that reaches the step is finite  -- the forward's NaN does not escape
    4. the padded batch counts no token          -- it does not dilute the step's denominator

The forward really does produce NaN: a mean over no supervised token is 0/0. What matters is that
the gradient was never NaN to begin with, and that ``ModelOutputLoss`` turns the reported value into
zero. Case 2 is run with the router's load-balancing loss both off and on, because that term is
computed from the routing rather than from the labels and is the one path that could carry a
gradient out of a batch with no labels.

PYTHONPATH carries the repository root, because a script's own directory is what Python puts on the
path and the editable install may point at another checkout:

    PYTHONPATH=. python examples/qwen3_vl_30b_perf/check_padded_work.py
"""

import sys
from types import SimpleNamespace
from typing import Any

import torch
from check_packing import build_model, make_document

from hyper_parallel.components.losses.model_output import ModelOutputLoss
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def micro_batch_of(config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one ``(model_inputs, loss_inputs)`` pair, split as the batch adapter splits it.

    Args:
        config: The model configuration, read for the patch geometry.

    Returns:
        The model inputs and the loss inputs of a single-row micro-batch carrying one image.
    """
    document = make_document(config, text_tokens=12, grid=[1, 2, 2], seed=1)
    row = {name: value.unsqueeze(0) for name, value in document.items()
           if name in ("input_ids", "labels", "mm_token_type_ids")}
    row["pixel_values"] = document["pixel_values"]
    row["image_grid_thw"] = document["image_grid_thw"]
    labels = row["labels"]
    model_inputs = dict(row)
    loss_inputs = {"labels": labels, "loss_mask": labels != IGNORE_INDEX}
    return model_inputs, loss_inputs


def gradients_of(model: Any, model_inputs: dict[str, Any], loss_inputs: dict[str, Any]) -> tuple[float, float]:
    """Run one forward-backward pass and report the loss and the largest gradient it produced.

    The loss is read through :class:`ModelOutputLoss`, which is what the trainer uses when no loss
    is configured, so the number returned is the one the step would see.

    Args:
        model: The model under test.
        model_inputs: Inputs forwarded to the model.
        loss_inputs: The loss-only fields, read for the labels.

    Returns:
        The reported loss and the largest absolute parameter gradient.
    """
    model.zero_grad(set_to_none=True)
    outputs = model(**model_inputs, use_cache=False)
    loss = ModelOutputLoss()(model_output=outputs, labels=loss_inputs["labels"])
    loss.backward()
    largest = max((float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None),
                  default=0.0)
    return float(loss.detach()), largest


def main() -> int:
    """Run the four checks and report.

    Returns:
        0 when the padded batch moved nothing, 1 otherwise.
    """
    failures = []

    model, config = build_model()
    model.train()
    model_inputs, loss_inputs = micro_batch_of(config)
    padded_inputs, padded_loss = VLMTrainer.padding_micro_batches(
        SimpleNamespace(base=None), [(model_inputs, loss_inputs)],
    )[0]

    # 1. The control: real work moves the weights, so a zero below means something.
    real_loss, real_grad = gradients_of(model, model_inputs, loss_inputs)
    print(f"real batch:                       loss = {real_loss:11.6f}   max|grad| = {real_grad:.6g}")
    if not real_grad > 0.0:
        failures.append("the real batch produced no gradient, so this check proves nothing")
    if not torch.isfinite(torch.tensor(real_loss)):
        failures.append(f"the real batch's loss is not finite: {real_loss}")

    # 2 and 3. Padded work, with the router's load-balancing term off and then on.
    for router in (False, True):
        config.text_config.output_router_logits = router
        model.config.text_config.output_router_logits = router
        padded_value, padded_grad = gradients_of(model, padded_inputs, padded_loss)
        label = f"padded batch, router aux {'on ' if router else 'off'}:"
        print(f"{label:34s}loss = {padded_value:11.6f}   max|grad| = {padded_grad:.6g}")
        if padded_grad != 0.0:
            failures.append(f"padded work moved the weights with router aux {router}: max|grad| = {padded_grad}")
        if not torch.isfinite(torch.tensor(padded_value)):
            failures.append(f"the padded batch's loss escaped as {padded_value} with router aux {router}")

    # 4. The padded batch must not dilute the step's token denominator either.
    real_tokens = int(count_loss_token(loss_inputs)["foundation_tokens"])
    padded_tokens = int(count_loss_token(padded_loss)["foundation_tokens"])
    print(f"supervised tokens:                real = {real_tokens:<11d}  padded = {padded_tokens}")
    if padded_tokens != 0:
        failures.append(f"padded work counted {padded_tokens} supervised tokens, so it would be weighed in")
    if real_tokens <= 0:
        failures.append("the real batch counted no supervised token, so this check proves nothing")

    print()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: padded work joins the step and leaves every gradient at zero")
    return 0


if __name__ == "__main__":
    sys.exit(main())
