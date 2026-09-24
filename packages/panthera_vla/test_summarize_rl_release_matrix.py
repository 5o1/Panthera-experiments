"""Regression tests for RL release-matrix TensorBoard aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

tensorboard = pytest.importorskip("tensorboard")
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

import summarize_rl_release_matrix as summary


def _write_scalar(root: Path, tag: str, value: float) -> None:
    event_dir = root / "tensorboard"
    event_dir.mkdir(parents=True)
    writer = EventFileWriter(str(event_dir))
    writer.add_event(
        Event(
            wall_time=1.0,
            step=0,
            summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]),
        )
    )
    writer.flush()
    writer.close()


def test_evaluation_matrix_includes_frozen_sft_r0(tmp_path: Path) -> None:
    for index, variant in enumerate(summary.EVAL_VARIANTS):
        _write_scalar(tmp_path / variant, "eval/success_once", index / 10.0)

    rows, scalars = summary._variant_rows(
        tmp_path,
        summary.EVAL_VARIANTS,
        tags=summary.EVAL_TAGS,
        allow_incomplete=False,
    )

    assert summary.VARIANTS == ("c0", "c1", "c2")
    assert summary.EVAL_VARIANTS == ("r0", "c0", "c1", "c2")
    assert "eval/panthera_invalid_release_once" in summary.EVAL_TAGS
    assert "eval/panthera_release_delay_actions" in summary.EVAL_TAGS
    assert (
        "eval/panthera_chunk_boundary_second_difference_l1_mean"
        in summary.EVAL_TAGS
    )
    assert [row["variant"] for row in rows] == list(summary.EVAL_VARIANTS)
    assert set(scalars) == set(summary.EVAL_VARIANTS)


def test_strict_eval_summary_rejects_missing_r0(tmp_path: Path) -> None:
    for variant in summary.VARIANTS:
        _write_scalar(tmp_path / variant, "eval/success_once", 0.0)

    with pytest.raises(FileNotFoundError, match="no TensorBoard event for r0"):
        summary._variant_rows(
            tmp_path,
            summary.EVAL_VARIANTS,
            tags=summary.EVAL_TAGS,
            allow_incomplete=False,
        )

    rows, _ = summary._variant_rows(
        tmp_path,
        summary.EVAL_VARIANTS,
        tags=summary.EVAL_TAGS,
        allow_incomplete=True,
    )
    assert [row["variant"] for row in rows] == list(summary.VARIANTS)
