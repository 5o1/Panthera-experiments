"""CPU-only contract tests for the release-arena evaluation launcher."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "pipelines/ci/evaluate_openvla_rl_release_gpu23.sh"


def _fake_project(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    project = tmp_path / "project"
    runtime = project / "runtime/rlinf-rl-gpu23"
    robotwin = project / "runtime/robotwin-rl-gpu23"
    python_bin = project / "envs/rlinf/bin/python"
    fake_bin = tmp_path / "bin"

    (runtime / "examples/embodiment/config").mkdir(parents=True)
    robotwin.mkdir(parents=True)
    python_bin.parent.mkdir(parents=True)
    fake_bin.mkdir()
    python_bin.symlink_to(Path(sys.executable).resolve())
    (runtime / "ASSEMBLY.json").write_text("{}\n", encoding="utf-8")
    (runtime / "examples/embodiment/config/robotwin_panthera_continuous_ppo.yaml").write_text(
        "runner: {}\n", encoding="utf-8"
    )
    (runtime / "examples/embodiment/train_embodied_agent.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env sh\nprintf '40000\\n'\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PANTHERA_LAB_ROOT": str(project),
            "PANTHERA_RL_RUN_STAMP": f"test-{tmp_path.name}",
            "WANDB_MODE": "offline",
        }
    )
    return project, env


def _run_launcher(
    tmp_path: Path, *arguments: str
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    project, env = _fake_project(tmp_path)
    output = tmp_path / "output"
    env["PANTHERA_RL_OUTPUT_DIR"] = str(output)
    ray_tmp = Path("/tmp") / f"prl-eval-{env['PANTHERA_RL_RUN_STAMP']}-{arguments[0]}"
    try:
        result = subprocess.run(
            [str(LAUNCHER), *arguments],
            cwd=REPO,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        shutil.rmtree(ray_tmp, ignore_errors=True)
    return result, output, project


def test_r0_is_explicit_checkpoint_free_frozen_sft_baseline(tmp_path: Path) -> None:
    result, output, _ = _run_launcher(tmp_path, "r0")

    assert result.returncode == 0, result.stderr
    config = json.loads((output / "eval-config.json").read_text(encoding="utf-8"))
    assert config["eval_label"] == "r0"
    assert config["policy_variant"] == "c0"
    assert config["frozen_sft_baseline"] is True
    assert config["checkpoint"] is None
    assert config["checkpoint_size"] is None
    assert config["checkpoint_sha256"] is None
    assert (output / "COMPLETE").is_file()
    assert (output / "exit-code.txt").read_text(encoding="utf-8").strip() == "0"


def test_rl_eval_requires_and_records_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "full_weights.pt"
    checkpoint.write_bytes(b"test-checkpoint")

    result, output, _ = _run_launcher(tmp_path, "c1", str(checkpoint))

    assert result.returncode == 0, result.stderr
    config = json.loads((output / "eval-config.json").read_text(encoding="utf-8"))
    assert config["eval_label"] == "c1"
    assert config["policy_variant"] == "c1"
    assert config["frozen_sft_baseline"] is False
    assert config["checkpoint"] == str(checkpoint.resolve())
    assert config["checkpoint_size"] == len(b"test-checkpoint")
    assert config["checkpoint_sha256"] == hashlib.sha256(
        b"test-checkpoint"
    ).hexdigest()
    assert (output / "COMPLETE").is_file()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("r0", "/tmp/unexpected.pt"), "不得传入权重路径"),
        (("c0",), "必须显式传入 full_weights.pt"),
        (("unknown",), "未知评测组"),
    ],
)
def test_eval_mode_and_checkpoint_contract_is_strict(
    arguments: tuple[str, ...], message: str
) -> None:
    result = subprocess.run(
        [str(LAUNCHER), *arguments],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
