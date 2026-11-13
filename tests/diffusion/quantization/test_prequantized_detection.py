# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Detection of pre-quantized checkpoints, which CPU offload loads straight onto the CPU."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.component_config import ComponentQuantizationConfig
from vllm_omni.quantization.factory import quant_config_is_serialized

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

AUTOROUND_INT4 = {"quant_method": "auto-round", "bits": 4, "group_size": 128, "sym": True, "data_type": "int"}
MODELOPT_FP8 = {"quant_method": "modelopt", "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN", "exclude_modules": []}


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        # Online methods requested with --quantization: never pre-quantized.
        ("fp8", False),
        ("int8", False),
        ("mxfp8", False),
        ("mxfp4", False),
        ("convrot_int8", None),  # no serialized flag: decided by where the config came from
        ("bitsandbytes", None),
        # Checkpoint metadata.
        (MODELOPT_FP8, True),
        ({"quant_method": "int8", "is_checkpoint_int8_serialized": True}, True),
        ({"quant_method": "mxfp8", "is_checkpoint_mxfp8_serialized": True}, True),
        ({"quant_method": "mxfp4", "is_checkpoint_mxfp4_serialized": True}, True),
        ({"quant_method": "fp8"}, False),  # method-only metadata: quantized online
    ],
)
def test_built_configs(spec, expected):
    assert quant_config_is_serialized(build_quant_config(spec)) is expected


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({}, None),
        ({"is_checkpoint_future_format_serialized": True}, True),  # the naming convention, no allowlist
        ({"is_checkpoint_future_format_serialized": False}, False),
        ({"is_checkpoint_serialized": True}, True),
        ({"is_checkpoint_fp8_serialized": False, "is_checkpoint_nvfp4_serialized": True}, True),
        ({"data_type": "mx_fp"}, True),  # AutoRound MXFP
        ({"is_checkpoint_fp8_serialized": "yes"}, None),  # only real booleans count
    ],
)
def test_naming_convention(attrs, expected):
    assert quant_config_is_serialized(SimpleNamespace(**attrs)) is expected


def test_component_configs():
    serialized = build_quant_config(MODELOPT_FP8)
    online = build_quant_config("fp8")
    unmarked = build_quant_config("convrot_int8")
    assert quant_config_is_serialized(ComponentQuantizationConfig({"transformer": serialized, "vae": None})) is True
    # Any online component still has to be quantized while loading.
    assert quant_config_is_serialized(ComponentQuantizationConfig({"transformer": serialized}, online)) is False
    assert quant_config_is_serialized(ComponentQuantizationConfig({"transformer": unmarked})) is None


@pytest.mark.parametrize(
    ("spec", "auto_detected", "expected"),
    [
        (MODELOPT_FP8, True, True),
        (MODELOPT_FP8, False, True),  # explicit --quantization modelopt on an FP8 checkpoint
        ("fp8", False, False),
        ({"quant_method": "fp8"}, True, False),  # the flag wins over provenance
        ("convrot_int8", False, False),
        ("convrot_int8", True, True),  # an unmarked config read from a checkpoint
        (AUTOROUND_INT4, True, True),  # AutoRound W4A16 checkpoint (unmarked, from config.json)
    ],
)
def test_loader_decision(spec, auto_detected, expected):
    config = build_quant_config(spec)
    assert DiffusersPipelineLoader._is_prequantized(config, auto_detected=auto_detected) is expected
