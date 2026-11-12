# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared CPU fixtures for the PAN 2 tests: a tiny transformer configuration, random checkpoints and inputs."""

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from vllm.config import set_current_vllm_config

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

DTYPE = torch.float32
TINY: dict[str, Any] = dict(
    attention_head_dim=16,
    context_mlp_ratio=2.0,
    in_channels=97,
    mlp_ratio=2.0,
    num_attention_heads=2,
    num_layers=2,
    num_refiner_layers=1,
    out_channels=48,
    patch_size=(1, 2, 2),
    refiner_mlp_ratio=2.0,
    rope_axes_dim=(4, 6, 6),
    rope_temporal_scale=2 / 3,
    rope_theta=256.0,
    text_embed_dim=32,
)


def random_checkpoint(root: Path) -> dict[str, torch.Tensor]:
    """Random weights for every TINY parameter, saved as a diffusers-layout transformer in ``root``."""
    from vllm_omni.diffusion.models.pan2.pan2_transformer import PAN2Transformer3DModel

    od_config = OmniDiffusionConfig.from_kwargs(dtype=DTYPE)
    od_config.model_config = {"pan2_host_modulation": False}
    ref = PAN2Transformer3DModel(od_config=od_config, **TINY)
    g = torch.Generator().manual_seed(0)
    state: dict[str, torch.Tensor] = {}
    for name, p in ref.named_parameters():
        # Undo the fused names so the checkpoint looks like the released diffusers layout.
        if ".to_qkv." in name or ".add_kv_proj." in name:
            fused, parts = (
                (".to_qkv.", ("to_q", "to_k", "to_v"))
                if ".to_qkv." in name
                else (
                    ".add_kv_proj.",
                    ("add_q_proj", "add_k_proj", "add_v_proj"),
                )
            )
            for chunk, part in zip(p.chunk(3, dim=0), parts):
                state[name.replace(fused, f".{part}.")] = torch.randn(chunk.shape, generator=g) * 0.05
            continue
        if ".ff.fc1." in name:
            name = name.replace(".ff.fc1.", ".ff.net.0.proj.")
        elif ".ff.fc2." in name:
            name = name.replace(".ff.fc2.", ".ff.net.2.")
        name = name.replace(".attn.to_out.", ".attn.to_out.0.") if name.startswith("transformer_blocks.") else name
        if name.startswith("context_refiner.") and ".attn.to_out." in name:
            name = name.replace(".attn.to_out.", ".attn.to_out.0.")
        state[name] = torch.randn(p.shape, generator=g) * 0.05
    (root / "transformer").mkdir(parents=True)
    save_file(
        {k: v.contiguous() for k, v in state.items()}, root / "transformer" / "diffusion_pytorch_model.safetensors"
    )
    return state


@pytest.fixture
def env(tmp_path, monkeypatch):
    # CPU-only: the diffusion process-group setup otherwise assumes an accelerator.
    from vllm_omni.diffusion.distributed import parallel_state

    monkeypatch.setattr(parallel_state.current_omni_platform, "get_device_count", lambda: 1, raising=False)
    monkeypatch.setattr(parallel_state.current_omni_platform, "set_device", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(parallel_state.current_omni_platform, "get_torch_device", lambda *a, **k: None, raising=False)
    # The tensors live on the CPU, so custom ops must take their native PyTorch path even when a GPU is visible.
    from vllm_omni.diffusion.layers.custom_op import CustomOp

    monkeypatch.setattr(CustomOp, "dispatch_forward", lambda self: self.forward_native)
    # Run with CUDA_VISIBLE_DEVICES="" on a CUDA host: report a capability instead of querying a device, so any
    # real CUDA call still fails loudly rather than touching a GPU.
    platform_cls = type(parallel_state.current_omni_platform)
    if platform_cls.__name__ == "CudaOmniPlatform" and not torch.cuda.is_available():
        monkeypatch.setattr(platform_cls, "get_device_capability", classmethod(lambda cls, device_id=0: None))
    od_config = OmniDiffusionConfig.from_kwargs(
        dtype=DTYPE, diffusion_attention_config={"default": "TORCH_SDPA"}, linear_backend="TORCH"
    )
    device = torch.device("cpu")
    vllm_config = create_diffusion_vllm_config(device, od_config)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{tmp_path / 'distributed'}",
        backend="gloo",
    )
    initialize_model_parallel(backend="gloo")
    try:
        with set_current_vllm_config(vllm_config), set_current_diffusion_config(od_config):
            yield tmp_path, od_config, vllm_config
    finally:
        destroy_distributed_env()


def tiny_inputs(batch: int, timesteps: list[float]):
    g = torch.Generator().manual_seed(1)
    return dict(
        hidden_states=torch.randn(batch, 97, 3, 4, 4, generator=g),
        timestep=torch.tensor(timesteps, dtype=torch.float32),
        encoder_hidden_states=torch.randn(batch, 5, 32, generator=g),
    )
