#!/usr/bin/env python3
"""Assemble a runtime tree from pinned upstream plus our overlay.

Our task environments, instructions, generated shard configs and audit scripts
were living inside the upstream checkouts: 415 untracked files and 9 modified
ones in RoboTwin alone, with zero local commits, so the only record of what we
had changed was a directory of loose diffs applied by shell.  "Which file is
ours" had no answer, and upgrading upstream meant reconciling a dirty tree.

Here the upstream checkout is read-only and stays git-clean, the overlay holds
everything we author, and the runtime is generated from the two.  Large
directories are symlinked so a runtime costs almost nothing; anything a patch
touches is copied first, because a patch must never reach back into the source.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PINNED = REPO / "externals/pinned.json"


class AssemblyError(RuntimeError):
    """The upstream is dirty, or a patch does not apply."""


@dataclass(frozen=True)
class Upstream:
    name: str
    overlay: Path
    # Files a patch modifies must exist as real copies in the runtime.
    copy_first: tuple[str, ...] = ()
    # Directories a run writes into.  A symlinked directory would send the
    # write through to the source: the collector generates an embodiment under
    # ``assets/embodiments``, and with ``assets`` linked wholesale that landed
    # in the 16 GB asset tree every runtime shares.  These are rebuilt as real
    # directories whose existing children are linked individually, so a new
    # entry stays in the runtime.
    writable: tuple[str, ...] = ()


UPSTREAMS = {
    "robotwin": Upstream(
        name="robotwin",
        overlay=REPO / "overlays/robotwin",
        copy_first=(
            "envs", "scripts", "env_cfg", "description",
        ),
        writable=("assets/embodiments",),
    ),
    "rlinf": Upstream(
        name="rlinf",
        overlay=REPO / "overlays/rlinf",
        copy_first=("rlinf", "requirements", "evaluations", "examples"),
    ),
}

# Where an overlay directory lands inside the runtime.  Anything not listed is
# copied to the same relative path.
PLACEMENT = {
    "robotwin": {
        "envs": "envs",
        "description": "description",
        # Upstream moved the task configs under env_cfg/ in the 223 commits
        # between 0008ae6 and 6dde571, along with script/ -> scripts/ and the
        # removal of the RLinf integration package.
        "task_config": "env_cfg/task_config",
        # Our embodiment and our object profiles belong beside upstream's
        # assets, not at the runtime root: RoboTwin looks for an embodiment
        # under ``assets/embodiments``, and placing ours at the root meant the
        # runtime never carried the source embodiment the collector derives
        # from.  It only surfaced once the pipeline actually ran from a
        # runtime instead of from the patched-in-place checkout.
        "assets": "assets",
    },
    "rlinf": {
        "config": "examples/embodiment/config",
        "evaluations": "evaluations/robotwin",
        "seeds": "seeds",
    },
}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def require_clean(source: Path) -> str:
    """Refuse to assemble from an upstream that has been written to."""
    if not (source / ".git").exists():
        raise AssemblyError(f"{source} is not a git checkout; cannot verify it is pristine")
    status = _git(source, "status", "--porcelain")
    if status:
        lines = status.splitlines()
        raise AssemblyError(
            f"{source} has {len(lines)} local change(s); the runtime is generated, "
            "so upstream must stay pristine. First offenders:\n  "
            + "\n  ".join(lines[:5])
        )
    return _git(source, "rev-parse", "HEAD")


def pinned_commit(name: str) -> str | None:
    """Return the declared upstream commit, if this is a real pinned upstream."""
    if not PINNED.is_file():
        return None
    payload = json.loads(PINNED.read_text(encoding="utf-8"))
    upstream = payload.get("upstreams", {}).get(name)
    return None if upstream is None else str(upstream["commit"])


def _link_or_copy(source: Path, target: Path, copy_names: set[str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if entry.name == ".git":
            continue
        destination = target / entry.name
        if entry.is_dir() and entry.name in copy_names:
            shutil.copytree(entry, destination, symlinks=True, dirs_exist_ok=True)
        elif entry.is_dir():
            destination.symlink_to(entry.resolve(), target_is_directory=True)
        else:
            shutil.copy2(entry, destination)


def _make_writable(source: Path, runtime: Path, relative: str) -> None:
    """Turn one path into a real directory whose children are linked."""
    current_source, current_runtime = source, runtime
    for part in Path(relative).parts:
        current_source = current_source / part
        if not current_source.is_dir():
            raise AssemblyError(f"{relative} is not a directory in {source}")
        target = current_runtime / part
        if target.is_symlink():
            target.unlink()
        target.mkdir(parents=True, exist_ok=True)
        for entry in sorted(current_source.iterdir()):
            child = target / entry.name
            if child.exists() or child.is_symlink():
                continue
            child.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
        current_runtime = target


def _apply_overlay(upstream: Upstream, runtime: Path) -> list[str]:
    placed: list[str] = []
    placement = PLACEMENT[upstream.name]
    for entry in sorted(upstream.overlay.iterdir()):
        if entry.name == "patches" or not entry.is_dir():
            continue
        relative = placement.get(entry.name, entry.name)
        destination = runtime if relative == "." else runtime / relative
        shutil.copytree(entry, destination, dirs_exist_ok=True)
        placed.append(f"{entry.name} -> {relative}")
    return placed


def _apply_patches(upstream: Upstream, runtime: Path) -> list[str]:
    patches = sorted((upstream.overlay / "patches").glob("*.patch"))
    applied: list[str] = []
    # Runtime normally lives below the Panthera checkout.  Without a ceiling,
    # `git apply` discovers the checkout's parent .git directory and resolves
    # patch paths against the source repository instead of `cwd=runtime`.
    # It can then return success while leaving the generated runtime unchanged.
    apply_environment = os.environ.copy()
    apply_environment["GIT_CEILING_DIRECTORIES"] = str(runtime.resolve().parent)
    for patch in patches:
        check = subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=runtime, capture_output=True, text=True, env=apply_environment,
        )
        if check.returncode != 0:
            reverse = subprocess.run(
                ["git", "apply", "--reverse", "--check", str(patch)],
                cwd=runtime, capture_output=True, text=True, env=apply_environment,
            )
            if reverse.returncode == 0:
                applied.append(f"{patch.name} (already present)")
                continue
            raise AssemblyError(
                f"{patch.name} does not apply to the runtime:\n{check.stderr.strip()}"
            )
        subprocess.run(
            ["git", "apply", str(patch)], cwd=runtime,
            check=True, env=apply_environment,
        )
        applied.append(patch.name)
    return applied


def assemble(name: str, source: Path, runtime: Path, clean: bool = True) -> dict:
    upstream = UPSTREAMS[name]
    commit = require_clean(source)
    expected = pinned_commit(name)
    if expected is not None and commit != expected:
        raise AssemblyError(
            f"{source} is at {commit}, but externals/pinned.json requires {expected}"
        )
    if clean and runtime.exists():
        shutil.rmtree(runtime)
    _link_or_copy(source, runtime, set(upstream.copy_first))
    # Before the overlay, not after: the overlay writes into some of these, and
    # a symlinked directory would send those writes into the shared source
    # rather than the runtime.  Assets are gitignored upstream, so the
    # git-clean check at the end would not have caught it either.
    for relative in upstream.writable:
        _make_writable(source, runtime, relative)
    placed = _apply_overlay(upstream, runtime)
    applied = _apply_patches(upstream, runtime)
    # Assembling must not have reached back into the source.
    after = _git(source, "status", "--porcelain")
    if after:
        raise AssemblyError(f"{source} was modified during assembly:\n{after}")
    record = {
        "upstream": name,
        "source": str(source),
        "commit": commit,
        "runtime": str(runtime),
        "overlay_placed": placed,
        "patches_applied": applied,
    }
    (runtime / "ASSEMBLY.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", choices=sorted(UPSTREAMS), required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="pinned upstream checkout; it is only read")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--keep", action="store_true",
                        help="update in place instead of rebuilding from scratch")
    cli = parser.parse_args()
    record = assemble(cli.upstream, cli.source.resolve(), cli.runtime.resolve(),
                      clean=not cli.keep)
    print(f"装配 {record['upstream']} @ {record['commit'][:12]} -> {record['runtime']}")
    for line in record["overlay_placed"]:
        print(f"  overlay  {line}")
    for line in record["patches_applied"]:
        print(f"  patch    {line}")
    print(f"  上游保持干净：{record['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
