import pytest

from render_rollout import inference_labels
from rollout import summarize_inference_timing


def test_inference_timing_is_amortized_over_executed_actions():
    timing = summarize_inference_timing(
        [
            {"row": 0, "inference_ms": 200.0},
            {"row": 20, "inference_ms": 300.0},
            {"row": 40, "inference_ms": 100.0},
        ],
        executed_actions=50,
        control_hz=50.0,
    )

    assert timing["query_ms"]["median"] == pytest.approx(200.0)
    assert timing["amortized_ms_per_action"]["median"] == pytest.approx(10.0)
    assert timing["realtime_load_x"]["median"] == pytest.approx(0.5)
    assert timing["deadline_miss_queries"] == 0
    assert "simulator rendering" in timing["excluded"]
    assert "lower-bound policy latency" in timing["real_workbench_interpretation"]


def test_video_labels_make_realtime_budget_and_scope_explicit():
    case = {
        "prediction_trace": [
            {"row": 0, "inference_ms": 40.0},
            {"row": 2, "inference_ms": 60.0},
        ]
    }

    labels = inference_labels(case, actions=3, control_period_ms=20.0)

    assert "20.0 ms/action" in labels[0][0]
    assert "1.00x OK" in labels[0][1]
    assert "3.00x LATE" in labels[2][1]
    assert "real-table lower bound" in labels[0][2]
    assert "render / physics / video excluded" in labels[0][3]
