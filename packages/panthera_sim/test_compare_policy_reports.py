import pytest

from compare_policy_reports import compare


def _report(backend, *, digest="same", episodes=(2, 4), success=1):
    return {
        "policy_backend": backend,
        "model": f"/{backend}",
        "dataset_digest": digest,
        "max_actions": 1000,
        "execution_horizon": 20,
        "temporal_ensemble": None,
        "total": len(episodes),
        "success": success,
        "gate_success": success,
        "by_posture": {},
        "cases": [
            {
                "episode": episode,
                "inference_timing": {
                    "amortized_ms_per_action": {"median": 8.0 + index}
                },
            }
            for index, episode in enumerate(episodes)
        ],
    }


def test_matching_reports_produce_paired_metrics():
    result = compare(_report("openvla-oft"), _report("openpi-pi05", success=2))
    assert result["comparison_contract"]["episodes"] == [2, 4]
    assert result["policies"][0]["success_rate"] == 0.5
    assert result["policies"][1]["success_rate"] == 1.0
    assert result["policies"][0]["median_amortized_ms_per_action"] == 8.5


def test_dataset_mismatch_is_rejected():
    with pytest.raises(ValueError, match="dataset_digest"):
        compare(_report("openvla-oft"), _report("openpi-pi05", digest="other"))


def test_episode_mismatch_is_rejected():
    with pytest.raises(ValueError, match="episodes"):
        compare(_report("openvla-oft"), _report("openpi-pi05", episodes=(2, 5)))
