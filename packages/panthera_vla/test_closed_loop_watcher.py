"""Artifact-safety checks for the asynchronous closed-loop validator."""

from pathlib import Path

from closed_loop_watcher import (
    SNAPSHOT_FILES,
    _checkpoint_stamp,
    _promote_success,
    _write_json_atomic,
)


def _checkpoint(root: Path) -> None:
    for relative in SNAPSHOT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())


def test_checkpoint_stamp_requires_every_deployable_component(tmp_path):
    run = tmp_path / "run"
    _checkpoint(run)
    before = _checkpoint_stamp(run)
    assert before is not None

    (run / "action_head--latest_checkpoint.pt").unlink()
    assert _checkpoint_stamp(run) is None


def test_checkpoint_stamp_detects_a_rewritten_component(tmp_path):
    run = tmp_path / "run"
    _checkpoint(run)
    before = _checkpoint_stamp(run)
    (run / "proprio_projector--latest_checkpoint.pt").write_bytes(b"new payload")
    assert _checkpoint_stamp(run) != before


def test_successful_snapshot_is_promoted_without_overwrite(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model.safetensors.index.json").write_text("{}")
    target = tmp_path / "closed-loop-best"

    assert _promote_success(candidate, target) == target
    assert not candidate.exists()
    assert (target / "model.safetensors.index.json").is_file()

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    try:
        _promote_success(replacement, target)
    except RuntimeError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("an accepted model must be immutable")


def test_success_sentinel_is_written_atomically(tmp_path):
    path = tmp_path / "run" / "CLOSED_LOOP_SUCCESS"
    _write_json_atomic(path, {"success": True, "model": "/model"})
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert not path.with_name(f".{path.name}.tmp").exists()
