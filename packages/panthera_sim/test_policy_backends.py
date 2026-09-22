import sys
import types

import numpy as np
import pytest

from policy_backends import OpenPI05Backend, create_backend


def test_openpi_backend_uses_public_observation_and_returns_7d(monkeypatch, tmp_path):
    calls = {}

    class FakePolicy:
        def infer(self, observation):
            calls["observation"] = observation
            return {"actions": np.ones((25, 7), dtype=np.float32)}

    policy_config = types.ModuleType("openpi.policies.policy_config")

    def create_trained_policy(config, checkpoint, pytorch_device=None):
        calls["config"] = config
        calls["checkpoint"] = checkpoint
        calls["device"] = pytorch_device
        return FakePolicy()

    policy_config.create_trained_policy = create_trained_policy
    policies = types.ModuleType("openpi.policies")
    policies.policy_config = policy_config
    openpi = types.ModuleType("openpi")
    openpi.policies = policies
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.policies", policies)
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", policy_config)

    custom_config = types.ModuleType("openpi_config")
    custom_config.build_train_config = lambda **kwargs: {"kwargs": kwargs}
    monkeypatch.setitem(sys.modules, "openpi_config", custom_config)

    backend = OpenPI05Backend(
        {
            "model": str(tmp_path / "checkpoint"),
            "openpi_repo_id": "panthera/schema10",
            "openpi_lora": True,
            "openpi_device": "cuda",
            "action_chunk": 25,
            "dataset_digest": "digest",
        }
    )
    actions = backend.predict(
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.zeros(7, dtype=np.float32),
        "insert cylinder",
    )
    assert actions.shape == (25, 7)
    assert calls["observation"]["observation/image"].shape == (24, 32, 3)
    assert calls["observation"]["observation/state"].shape == (7,)
    assert calls["device"] == "cuda"
    assert calls["config"]["kwargs"]["source_dataset_digest"] == "digest"


def test_backend_rejects_wrong_openpi_output_shape(monkeypatch, tmp_path):
    class FakePolicy:
        def infer(self, observation):
            return {"actions": np.zeros((25, 32), dtype=np.float32)}

    policy_config = types.ModuleType("openpi.policies.policy_config")
    policy_config.create_trained_policy = lambda *args, **kwargs: FakePolicy()
    policies = types.ModuleType("openpi.policies")
    policies.policy_config = policy_config
    openpi = types.ModuleType("openpi")
    openpi.policies = policies
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.policies", policies)
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", policy_config)
    custom_config = types.ModuleType("openpi_config")
    custom_config.build_train_config = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "openpi_config", custom_config)

    backend = OpenPI05Backend(
        {
            "model": str(tmp_path / "checkpoint"),
            "openpi_repo_id": "panthera/schema10",
            "openpi_lora": False,
            "openpi_device": None,
            "action_chunk": 25,
            "dataset_digest": "digest",
        }
    )
    with pytest.raises(ValueError, match=r"\[N, 7\]"):
        backend.predict(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros(7), "task")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown policy backend"):
        create_backend({"policy_backend": "not-a-policy"})
