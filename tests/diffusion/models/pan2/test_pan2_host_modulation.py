# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PAN 2 host modulation: the norm1.linear modulation computed off the accelerator must match the in-model path."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.diffusion.models.pan2.conftest import TINY, random_checkpoint, tiny_inputs
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.pan2.host_modulation import (
    PAN2HostModulation,
    host_modulation_enabled,
    is_host_modulation_weight,
    mmap_modulation_tensors,
    modulation_bytes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Optional check against the released checkpoint: set PAN2_REAL_ROOT to a local copy of IFM/PAN2.
REAL_ROOT = Path(os.environ["PAN2_REAL_ROOT"]) if os.environ.get("PAN2_REAL_ROOT") else None


def _build(od_config, root: Path, host: bool, state: dict[str, torch.Tensor]):
    from vllm_omni.diffusion.models.pan2.pan2_transformer import PAN2Transformer3DModel

    od_config.model = str(root)
    od_config.model_config = (
        {"pan2_host_modulation": True, "pan2_host_modulation_threads": 2} if host else {"pan2_host_modulation": False}
    )
    model = PAN2Transformer3DModel(od_config=od_config, **TINY).eval()
    model.load_weights(list(state.items()))
    return model


def test_host_model_has_no_modulation_linear(env):
    tmp, od_config, _ = env
    state = random_checkpoint(tmp / "m")
    host = _build(od_config, tmp / "m", True, state)
    base = _build(od_config, tmp / "m", False, state)
    assert not any("norm1.linear" in n for n, _ in host.named_parameters())
    assert sum("norm1.linear" in n for n, _ in base.named_parameters()) == 2 * TINY["num_layers"]
    assert host.host_modulation_skipped_weights == 2 * TINY["num_layers"]
    # Everything else is identical: the flag only removes the modulation linears.
    base_names = {n for n, _ in base.named_parameters() if "norm1.linear" not in n}
    assert base_names == {n for n, _ in host.named_parameters()}


@pytest.mark.parametrize("timesteps", [[937.5, 937.5], [500.0, 500.0], [0.0, 0.0], [999.0, 250.0]])
def test_forward_matches_in_model_modulation(env, timesteps):
    tmp, od_config, vllm_config = env
    state = random_checkpoint(tmp / "m")
    base = _build(od_config, tmp / "m", False, state)
    host = _build(od_config, tmp / "m", True, state)
    inputs = tiny_inputs(len(timesteps), timesteps)
    with torch.inference_mode(), set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config):
        expected = base(**inputs)
        actual = host(**inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_prefetch_then_hits(env):
    tmp, od_config, vllm_config = env
    state = random_checkpoint(tmp / "m")
    host = _build(od_config, tmp / "m", True, state)
    schedule = torch.tensor([999.0, 900.0, 750.0, 10.0])
    host.prefetch_host_modulation(schedule)
    assert host.host_modulation.stats["computed"] == 4
    with torch.inference_mode(), set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config):
        for t in schedule.tolist():
            host(**tiny_inputs(2, [t, t]))
    assert host.host_modulation.stats["misses"] == 0
    assert host.host_modulation.stats["hits"] == 4
    host.prefetch_host_modulation(schedule)  # fully cached: nothing recomputed
    assert host.host_modulation.stats["computed"] == 4


def test_cache_eviction_is_bounded():
    dim, blocks = 8, 3
    weights = [(torch.randn(6 * dim, dim), torch.randn(6 * dim)) for _ in range(blocks)]
    hm = PAN2HostModulation(weights, num_blocks=blocks, dim=dim, cache_mb=1)
    entry = blocks * 6 * dim * 4
    assert hm._max_entries == (2**20) // entry
    temb = torch.randn(1, dim)
    for t in range(hm._max_entries + 5):
        hm.lookup(torch.tensor([float(t)]), temb)
    assert len(hm._cache) == hm._max_entries


def test_compute_matches_linear():
    dim, blocks = 16, 4
    g = torch.Generator().manual_seed(3)
    weights = [(torch.randn(6 * dim, dim, generator=g), torch.randn(6 * dim, generator=g)) for _ in range(blocks)]
    hm = PAN2HostModulation(weights, num_blocks=blocks, dim=dim)
    temb = torch.randn(5, dim, generator=g)
    out = hm.compute(temb)
    for i, (w, b) in enumerate(weights):
        torch.testing.assert_close(out[:, i], torch.nn.functional.linear(torch.nn.functional.silu(temb), w, b))


def test_mmap_reader_matches_safe_open(tmp_path):
    dim, blocks = 8, 2
    state = {}
    for i in range(blocks):
        state[f"transformer_blocks.{i}.norm1.linear.weight"] = torch.randn(6 * dim, dim)
        state[f"transformer_blocks.{i}.norm1.linear.bias"] = torch.randn(6 * dim)
        state[f"transformer_blocks.{i}.attn.to_q.weight"] = torch.randn(dim, dim).bfloat16()
    save_file(dict(list(state.items())[:3]), tmp_path / "a.safetensors")
    save_file(dict(list(state.items())[3:]), tmp_path / "b.safetensors")
    mapped = mmap_modulation_tensors(tmp_path, blocks)
    for i, (w, b) in enumerate(mapped):
        assert torch.equal(w, state[f"transformer_blocks.{i}.norm1.linear.weight"])
        assert torch.equal(b, state[f"transformer_blocks.{i}.norm1.linear.bias"])


def test_mmap_reader_rejects_non_fp32(tmp_path):
    save_file(
        {
            "transformer_blocks.0.norm1.linear.weight": torch.randn(48, 8).bfloat16(),
            "transformer_blocks.0.norm1.linear.bias": torch.randn(48),
        },
        tmp_path / "a.safetensors",
    )
    with pytest.raises(ValueError, match="F32"):
        mmap_modulation_tensors(tmp_path, 1)


def test_weight_name_filter():
    assert is_host_modulation_weight("transformer_blocks.43.norm1.linear.weight")
    assert is_host_modulation_weight("transformer_blocks.0.norm1.linear.bias")
    assert not is_host_modulation_weight("transformer_blocks.0.norm1_context.weight")
    assert not is_host_modulation_weight("norm_out.linear.weight")
    assert not is_host_modulation_weight("context_refiner.0.norm1.weight")


@pytest.mark.skipif(
    REAL_ROOT is None or not (REAL_ROOT / "transformer").is_dir(), reason="set PAN2_REAL_ROOT to a local IFM/PAN2"
)
def test_real_checkpoint_mmap_matches_safe_open():
    assert REAL_ROOT is not None
    transformer = REAL_ROOT / "transformer"
    mapped = mmap_modulation_tensors(transformer, 44)
    import json

    weight_map = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    for i in (0, 21, 43):
        name = f"transformer_blocks.{i}.norm1.linear.weight"
        with safe_open(transformer / weight_map[name], framework="pt") as f:
            ref = f.get_slice(name)[:64]
        assert torch.equal(mapped[i][0][:64], ref)
        assert mapped[i][0].shape == (24576, 4096)


# Released checkpoint: 44 blocks, dim 4096 -> 17.7 GB of FP32 modulation per tensor-parallel rank.
RELEASED = dict(num_blocks=44, dim=4096)
GB = 10**9


def test_released_modulation_size():
    assert round(modulation_bytes(**RELEASED) / GB, 1) == 17.7


@pytest.mark.parametrize(
    ("model_config", "device_memory", "streams_blocks", "expected"),
    [
        ({}, 24 * GB, False, True),  # RTX 4090 / 3090: auto turns it on
        ({}, 32 * GB, False, True),  # RTX 5090
        ({}, 48 * GB, False, True),  # 48 GB GPU without layerwise offload
        # The existing PAN 2 recipes keep the original path:
        ({}, 48 * GB, True, False),  # PAN2-L40S (layerwise offload)
        ({}, 80 * GB, True, False),  # PAN2-H100, 1 GPU (layerwise offload)
        ({}, 80 * GB, False, False),  # PAN2-H100, 8 GPUs
        ({}, 96 * GB, False, False),  # PAN2-RTX-PRO-6000, 2 GPUs
        ({}, 141 * GB, False, False),  # PAN2 on H200 and larger
        ({}, None, False, False),  # no accelerator or unknown memory
        ({"pan2_host_modulation": "auto"}, 24 * GB, False, True),
        ({"pan2_host_modulation": False}, 24 * GB, False, False),  # explicit values always win
        ({"pan2_host_modulation": True}, 141 * GB, False, True),
        ({"pan2_host_modulation": True}, 48 * GB, True, True),
        ({"pan2_host_modulation": True}, None, False, True),
    ],
)
def test_host_modulation_auto_resolution(model_config, device_memory, streams_blocks, expected):
    od_config = SimpleNamespace(model_config=model_config)
    got = host_modulation_enabled(od_config, device_memory=device_memory, streams_blocks=streams_blocks, **RELEASED)
    assert got is expected


@pytest.mark.parametrize("value", ["on", "yes", 1, None])
def test_host_modulation_rejects_unknown_values(value):
    od_config = SimpleNamespace(model_config={"pan2_host_modulation": value})
    with pytest.raises(ValueError, match="pan2_host_modulation"):
        host_modulation_enabled(od_config, device_memory=24 * GB, streams_blocks=False, **RELEASED)
