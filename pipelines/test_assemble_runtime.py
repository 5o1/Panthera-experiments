"""Checks for the runtime assembler.

The state it exists to prevent is the one measured on 2026-09-19: 415 untracked
and 9 modified files inside the RoboTwin checkout with zero local commits, so
"what have we changed" had no answer and upgrading upstream meant reconciling a
dirty tree.  The guarantees tested here are that upstream is never written to,
and that a dirty upstream is refused rather than silently used.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from assemble_runtime import (
    PINNED,
    PLACEMENT,
    REPO,
    UPSTREAMS,
    AssemblyError,
    assemble,
    require_clean,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def upstream(tmp_path: Path, monkeypatch) -> Path:
    source = tmp_path / "source"
    (source / "envs").mkdir(parents=True)
    (source / "big_assets").mkdir()
    (source / "envs" / "_base_task.py").write_text("STEP_LIMIT = 1000\n")
    (source / "big_assets" / "mesh.bin").write_bytes(b"\x00" * 1024)
    (source / "README.md").write_text("upstream\n")
    _git(source, "init", "-q")
    _git(source, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(source, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")

    overlay = tmp_path / "overlay"
    (overlay / "envs").mkdir(parents=True)
    (overlay / "patches").mkdir()
    (overlay / "envs" / "our_task.py").write_text("TASK = 'panthera'\n")
    monkeypatch.setitem(
        UPSTREAMS, "demo",
        UPSTREAMS["robotwin"].__class__(name="demo", overlay=overlay, copy_first=("envs",)),
    )
    monkeypatch.setitem(PLACEMENT, "demo", {"envs": "envs"})
    return source


def test_clean_upstream_assembles(tmp_path, upstream):
    runtime = tmp_path / "runtime"
    record = assemble("demo", upstream, runtime)
    assert (runtime / "envs" / "our_task.py").read_text() == "TASK = 'panthera'\n"
    assert (runtime / "envs" / "_base_task.py").is_file()
    assert record["commit"]
    assert (runtime / "ASSEMBLY.json").is_file()


def test_real_upstream_manifest_pins_the_migrated_robotwin_commit():
    import json

    payload = json.loads(PINNED.read_text(encoding="utf-8"))
    robotwin = payload["upstreams"]["robotwin"]
    assert robotwin["branch"] == "main"
    assert robotwin["commit"] == "6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755"
    assert robotwin["directory"] == "RoboTwin"


def test_large_directories_are_linked_not_copied(tmp_path, upstream):
    runtime = tmp_path / "runtime"
    assemble("demo", upstream, runtime)
    # envs is copied because a patch could touch it; assets are linked.
    assert (runtime / "big_assets").is_symlink()
    assert not (runtime / "envs").is_symlink()


def test_upstream_is_untouched_by_assembly(tmp_path, upstream):
    assemble("demo", upstream, tmp_path / "runtime")
    status = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    assert status.stdout == ""
    assert not (upstream / "envs" / "our_task.py").exists()


def test_dirty_upstream_is_refused_and_counted(tmp_path, upstream):
    (upstream / "envs" / "stray.py").write_text("ours, in the wrong tree\n")
    with pytest.raises(AssemblyError, match="local change"):
        assemble("demo", upstream, tmp_path / "runtime")


def test_modified_upstream_file_is_refused(tmp_path, upstream):
    (upstream / "envs" / "_base_task.py").write_text("STEP_LIMIT = 3200\n")
    with pytest.raises(AssemblyError, match="local change"):
        assemble("demo", upstream, tmp_path / "runtime")


def test_a_non_git_source_cannot_be_verified(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(AssemblyError, match="not a git checkout"):
        require_clean(plain)


def test_patch_applies_to_the_runtime_only(tmp_path, upstream):
    overlay = UPSTREAMS["demo"].overlay
    (overlay / "patches" / "01_step_limit.patch").write_text(
        "diff --git a/envs/_base_task.py b/envs/_base_task.py\n"
        "--- a/envs/_base_task.py\n"
        "+++ b/envs/_base_task.py\n"
        "@@ -1 +1 @@\n"
        "-STEP_LIMIT = 1000\n"
        "+STEP_LIMIT = 3200\n"
    )
    runtime = tmp_path / "runtime"
    record = assemble("demo", upstream, runtime)
    assert "01_step_limit.patch" in record["patches_applied"]
    assert (runtime / "envs" / "_base_task.py").read_text() == "STEP_LIMIT = 3200\n"
    # The source keeps the value it had.
    assert (upstream / "envs" / "_base_task.py").read_text() == "STEP_LIMIT = 1000\n"


def test_patch_target_stays_runtime_when_runtime_is_inside_this_checkout(upstream):
    """Lab keeps generated runtimes below the Panthera Git checkout."""
    overlay = UPSTREAMS["demo"].overlay
    (overlay / "patches" / "01_step_limit.patch").write_text(
        "diff --git a/envs/_base_task.py b/envs/_base_task.py\n"
        "--- a/envs/_base_task.py\n"
        "+++ b/envs/_base_task.py\n"
        "@@ -1 +1 @@\n"
        "-STEP_LIMIT = 1000\n"
        "+STEP_LIMIT = 3200\n"
    )
    (REPO / "var").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="assembly-test-", dir=REPO / "var") as root:
        runtime = Path(root) / "runtime"
        assemble("demo", upstream, runtime)
        assert (runtime / "envs" / "_base_task.py").read_text() == "STEP_LIMIT = 3200\n"


def test_a_patch_that_does_not_apply_fails_loudly(tmp_path, upstream):
    overlay = UPSTREAMS["demo"].overlay
    (overlay / "patches" / "01_wrong.patch").write_text(
        "diff --git a/envs/_base_task.py b/envs/_base_task.py\n"
        "--- a/envs/_base_task.py\n"
        "+++ b/envs/_base_task.py\n"
        "@@ -1 +1 @@\n"
        "-SOMETHING_ELSE = 1\n"
        "+SOMETHING_ELSE = 2\n"
    )
    with pytest.raises(AssemblyError, match="does not apply"):
        assemble("demo", upstream, tmp_path / "runtime")


def test_assembly_record_names_what_was_placed(tmp_path, upstream):
    record = assemble("demo", upstream, tmp_path / "runtime")
    assert record["overlay_placed"] == ["envs -> envs"]
    assert record["upstream"] == "demo"


@pytest.mark.parametrize("upstream_name", ("robotwin", "rlinf", "openvla"))
def test_no_two_patches_touch_the_same_file(upstream_name):
    """One patch per upstream file, or they stop applying.

    The patches were organised by feature, and three of them changed
    ``script/collect_data.py``.  Each was diffed against a tree where the other
    two were already applied, so each carried context the others removed and
    ``git apply`` refused the second one: assembling the runtime failed with
    "patch does not apply" on a patch set that had been in the repository for
    days, because nothing ever assembled from a clean upstream to find out.
    A file's whole change now lives in one patch, and the features it carries
    are named in its header.
    """
    import re

    owners: dict[str, list[str]] = {}
    patch_root = REPO / "overlays" / upstream_name / "patches"
    for patch in sorted(patch_root.glob("*.patch")):
        text = patch.read_text(encoding="utf-8", errors="replace")
        for path in re.findall(r"^\+\+\+ b/(.*)$", text, re.M):
            owners.setdefault(path, []).append(patch.name)
    shared = {path: names for path, names in owners.items() if len(names) > 1}
    assert shared == {}, (
        "these upstream files are changed by more than one patch, which makes "
        "the later ones fail to apply: "
        + "; ".join(f"{path} <- {', '.join(names)}" for path, names in shared.items())
    )


def test_a_patched_file_is_not_also_shipped_whole():
    """The same change must not live in both a patch and an overlay file.

    Three ``task_config`` files were carried by the overlay in full *and* by a
    patch generated from the same tree.  The assembly still worked -- the patch
    reverse-applied and was reported "already present" -- so the duplication
    was invisible, and a later edit to one copy would have silently diverged
    from the other.
    """
    import re

    overlay = REPO / "overlays/robotwin"
    shipped = {
        path.relative_to(overlay).as_posix()
        for path in overlay.rglob("*")
        if path.is_file() and "patches/" not in path.relative_to(overlay).as_posix()
    }
    patched = set()
    for patch in sorted((overlay / "patches").glob("*.patch")):
        text = patch.read_text(encoding="utf-8", errors="replace")
        patched.update(re.findall(r"^\+\+\+ b/(.*)$", text, re.M))
    both = sorted(shipped & patched)
    assert both == [], (
        "these are both shipped whole and patched: " + ", ".join(both)
    )


def test_a_writable_path_does_not_reach_back_into_the_source(tmp_path, upstream, monkeypatch):
    """A run must be able to add to it without touching the shared source.

    The collector generates an embodiment under ``assets/embodiments``.  With
    ``assets`` symlinked wholesale -- it is 16 GB and shared by every runtime --
    that write went through the link and into the source tree.
    """
    stock = upstream / "assets" / "embodiments" / "stock"
    stock.mkdir(parents=True)
    (stock / "config.yml").write_text("a: 1\n")
    (upstream / "assets" / "huge").mkdir()
    (upstream / "assets" / "huge" / "mesh.bin").write_bytes(b"\x00" * 16)
    _git(upstream, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(upstream, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "assets")

    demo = UPSTREAMS["demo"]
    monkeypatch.setitem(
        UPSTREAMS, "demo",
        demo.__class__(name="demo", overlay=demo.overlay, copy_first=demo.copy_first,
                       writable=("assets/embodiments",)),
    )

    runtime = tmp_path / "runtime"
    assemble("demo", upstream, runtime)

    embodiments = runtime / "assets" / "embodiments"
    assert not embodiments.is_symlink(), "the writable path is still a link"
    assert (embodiments / "stock").is_symlink(), "existing entries should be linked"
    assert (embodiments / "stock" / "config.yml").read_text() == "a: 1\n"

    (embodiments / "generated").mkdir()
    assert not (upstream / "assets" / "embodiments" / "generated").exists(), (
        "a write to the runtime reached the source"
    )
    assert (runtime / "assets" / "huge").is_symlink(), (
        "directories that are not written to should still be linked"
    )
