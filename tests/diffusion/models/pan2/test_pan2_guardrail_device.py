# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PAN 2 guardrail placement: ``model_config.guardrail_device = "cpu"`` keeps the guards off the accelerator."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.pan2 import pipeline_pan2

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeGuard:
    def __init__(self, moves: list):
        self._moves = moves

    def to(self, device):
        self._moves.append(str(device))
        return self


class _FakeChecker:
    def __init__(self):
        self.moves: list[str] = []
        self.text_guard = _FakeGuard(self.moves)
        self.video_guard = _FakeGuard(self.moves)

    def to(self, device):
        self.moves.append(f"checker->{device}")
        return self

    def check_text_safety(self, prompt):
        return True

    def check_video_safety(self, frames):
        return frames


@pytest.fixture
def fresh_guardrails(monkeypatch):
    monkeypatch.setattr(pipeline_pan2, "_text_guardrail", None)
    monkeypatch.setattr(pipeline_pan2, "_video_guardrail", None)
    checker = _FakeChecker()
    monkeypatch.setattr(pipeline_pan2.PAN2SafetyChecker, "from_pretrained", staticmethod(lambda: checker))
    return checker


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        ({}, {"offload_to_cpu": False, "run_on_cpu": False}),
        ({"offload_guardrail_models": True}, {"offload_to_cpu": True, "run_on_cpu": False}),
        ({"guardrail_device": "cpu"}, {"offload_to_cpu": False, "run_on_cpu": True}),
        ({"guardrail_device": "CPU", "offload_guardrail_models": True}, {"offload_to_cpu": True, "run_on_cpu": True}),
    ],
)
def test_guardrail_device_reaches_initializer(monkeypatch, model_config, expected):
    seen = {}
    monkeypatch.setattr(pipeline_pan2, "is_guardrails_enabled", lambda od_config: True)
    monkeypatch.setattr(pipeline_pan2, "_init_default_guardrails", lambda **kw: seen.update(kw))
    pipeline_pan2._ensure_guardrails_initialized(SimpleNamespace(model_config=model_config))
    assert seen == expected


def test_invalid_guardrail_device_is_rejected(monkeypatch):
    monkeypatch.setattr(pipeline_pan2, "is_guardrails_enabled", lambda od_config: True)
    with pytest.raises(ValueError, match="guardrail_device"):
        pipeline_pan2._ensure_guardrails_initialized(SimpleNamespace(model_config={"guardrail_device": "gpu1"}))


@pytest.mark.parametrize("offload_to_cpu", [False, True])
def test_cpu_guardrails_never_move_to_the_accelerator(fresh_guardrails, offload_to_cpu):
    pipeline_pan2._init_default_guardrails(offload_to_cpu=offload_to_cpu, run_on_cpu=True)
    pipeline_pan2._check_text_safety("a structured caption")
    pipeline_pan2._video_guardrail([0])
    assert fresh_guardrails.moves == []


def test_offloaded_guardrails_still_move_per_check(fresh_guardrails, monkeypatch):
    monkeypatch.setattr(pipeline_pan2.current_omni_platform, "empty_cache", lambda: None, raising=False)
    pipeline_pan2._init_default_guardrails(offload_to_cpu=True, run_on_cpu=False)
    pipeline_pan2._check_text_safety("a structured caption")
    assert fresh_guardrails.moves[-1] == "cpu" and len(fresh_guardrails.moves) == 2
