# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PAN 2 joint attention hands contiguous token splits to its output projections.

At batch > 1, splitting ``[video; text]`` along the token axis yields strided views. Quantized linears such as the
FP8 scaled-mm kernels flatten their input with ``.view()`` and fail on them, so the split must be made contiguous.
"""

import pytest
import torch

from tests.diffusion.models.pan2.conftest import TINY, random_checkpoint, tiny_inputs
from vllm_omni.diffusion.forward_context import set_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("batch", [1, 2])
def test_output_projections_receive_contiguous_inputs(env, batch):
    from vllm_omni.diffusion.models.pan2.pan2_transformer import PAN2Transformer3DModel

    tmp, od_config, vllm_config = env
    state = random_checkpoint(tmp / "m")
    od_config.model = str(tmp / "m")
    od_config.model_config = {"pan2_host_modulation": False}
    model = PAN2Transformer3DModel(od_config=od_config, **TINY).eval()
    model.load_weights(list(state.items()))

    seen: list[tuple[str, bool]] = []

    def record(name):
        def hook(module, args):
            seen.append((name, args[0].is_contiguous()))

        return hook

    for i, block in enumerate(model.transformer_blocks):
        block.attn.to_out.register_forward_pre_hook(record(f"{i}.to_out"))
        block.attn.to_add_out.register_forward_pre_hook(record(f"{i}.to_add_out"))

    with torch.inference_mode(), set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config):
        model(**tiny_inputs(batch, [500.0] * batch))

    assert len(seen) == 2 * TINY["num_layers"]
    assert all(contiguous for _, contiguous in seen), [name for name, contiguous in seen if not contiguous]
