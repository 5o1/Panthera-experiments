"""Repository-level guards for the 2026-09-20 layout closure."""

from __future__ import annotations

from pathlib import Path
import re

import pytest


REPO = Path(__file__).resolve().parents[1]
REPO_ROLE_FILE = REPO / ".panthera-repo-role"
REPO_ROLE = (
    REPO_ROLE_FILE.read_text(encoding="utf-8").strip()
    if REPO_ROLE_FILE.exists()
    else "source"
)
CURRENT_ENTRYPOINTS = (
    REPO / "tools/activate_lab_vla.sh",
    REPO / "tools/bootstrap_lab_vla.sh",
    REPO / "tools/bootstrap_lab_panthera_embodiment.sh",
    REPO / "tools/bootstrap_lab_panthera_phone_scene.sh",
    REPO / "tools/run_lab_robotwin_panthera_v2_pilot.sh",
    REPO / "pipelines/ci/gate_expert_replay.sh",
    REPO / "pipelines/ci/gate_single_trajectory_overfit.sh",
)


@pytest.mark.parametrize("path", CURRENT_ENTRYPOINTS, ids=lambda path: path.name)
def test_current_lab_entrypoints_do_not_use_the_old_runtime_layout(path: Path):
    text = path.read_text(encoding="utf-8")
    forbidden = (
        "${workspace}/RoboTwin",
        "${workspace}/RLinf",
        "${robotwin_root}/task_config",
        "${robotwin_root}/script/",
        "python script/collect_data.py",
        "${workspace}/activate_lab_vla.sh",
        "${workspace}/bin/activate_lab_vla.sh",
    )
    found = [token for token in forbidden if token in text]
    assert found == [], f"{path.relative_to(REPO)} still uses {found}"


def test_active_lab_scripts_use_the_tracked_activation_entrypoint():
    offenders = []
    for path in sorted((REPO / "tools").glob("*.sh")):
        if "${workspace}/bin/activate_lab_vla.sh" in path.read_text(
            encoding="utf-8"
        ):
            offenders.append(str(path.relative_to(REPO)))
    for path in sorted((REPO / "pipelines").rglob("*.sh")):
        if "${workspace}/bin/activate_lab_vla.sh" in path.read_text(
            encoding="utf-8"
        ):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == []


def test_lab_bootstrap_does_not_depend_on_the_nas_mount():
    text = (REPO / "tools/bootstrap_lab_vla.sh").read_text(encoding="utf-8")
    assert "/mnt/hulab" not in text
    assert "PANTHERA_NAS_ROOT" not in text
    assert "experiment_notebooks" not in text


def test_lab_bootstrap_locates_openvla_without_importing_it_before_patch():
    text = (REPO / "tools/bootstrap_lab_vla.sh").read_text(encoding="utf-8")
    assert 'distribution("openvla-oft").locate_file("")' in text
    assert "from experiments.robot import openvla_utils" not in text
    assert 'openvla_git_ceiling=$(dirname "$openvla_root")' in text
    assert 'GIT_CEILING_DIRECTORIES="$openvla_git_ceiling"' in text


def test_primary_context_documents_link_the_latest_decisions():
    if REPO_ROLE == "gpu_node":
        pytest.skip("technical reports are owned by the WSL source repository")
    expected = {
        "docs/18_eval_harness_defects_2026-09-19.md",
        "docs/19_architecture_plan_2026-09-19.md",
        "docs/20_eval_observation_mismatch_2026-09-19.md",
        "docs/21_upstream_migration_2026-09-20.md",
        "docs/22_single_trajectory_overfit_gate_hardening_2026-09-20.md",
    }
    for document in (REPO / "README.md", REPO / "AGENTS.md"):
        text = document.read_text(encoding="utf-8")
        missing = sorted(path for path in expected if path not in text)
        assert missing == [], f"{document.name} omits {missing}"


def test_repository_role_keeps_technical_reports_on_wsl_only():
    assert REPO_ROLE in {"source", "gpu_node"}
    if REPO_ROLE == "gpu_node":
        assert not (REPO / "docs").exists(), (
            "gpu_node must not contain technical reports; keep them in the WSL "
            "Panthera-experiments repository"
        )
    else:
        assert (REPO / "docs").is_dir(), "source repository must retain docs/"


def test_lab_mirror_does_not_deploy_repository_specific_context():
    text = (REPO / "tools/lab_mirror.sh").read_text(encoding="utf-8")
    match = re.search(r"^paths=\(([^)]*)\)$", text, flags=re.MULTILINE)
    assert match is not None
    deployed = set(match.group(1).split())
    assert deployed.isdisjoint(
        {"docs", "README.md", "AGENTS.md", "CLAUDE.md", ".gitignore"}
    )


@pytest.mark.parametrize("document_name", ("README.md", "AGENTS.md", "CLAUDE.md"))
def test_primary_context_document_local_links_exist(document_name: str):
    document = REPO / document_name
    text = document.read_text(encoding="utf-8")
    missing = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = target.strip("<>")
        if "://" in target or target.startswith("#"):
            continue
        relative = target.split("#", 1)[0].replace("%20", " ")
        if relative and not (document.parent / relative).exists():
            missing.append(target)
    assert missing == [], f"{document.name} has missing local links: {missing}"


def test_single_trajectory_gate_preserves_and_names_its_actual_training_data():
    text = (REPO / "pipelines/ci/gate_single_trajectory_overfit.sh").read_text(
        encoding="utf-8"
    )
    assert 'export PANTHERA_ACTION_CHUNK="$action_chunk"' in text
    assert 'export PANTHERA_SOURCE_DATASET_ROOT="$subset"' in text
    assert '--dataset-root "$subset"' in text
    assert 'rm -rf "$root"' not in text
    assert 'keep="${CI_OVERFIT_KEEP:-1}"' in text
    assert '--external_stop_file "$sentinel"' in text
    assert 'max_steps="${CI_OVERFIT_MAX_STEPS:-40000}"' in text
    assert '--max_steps "$max_steps"' in text
    assert '"max_steps": int(max_steps)' in text


def test_openvla_patch_keeps_the_hard_budget_with_external_closed_loop_stopping():
    text = (
        REPO / "overlays/openvla/patches/openvla_oft_finetune_and_ddp.patch"
    ).read_text(encoding="utf-8")
    assert 'external_stop_file: Optional[str]' in text
    assert 'stop_reason = "external_success"' in text
    assert 'if log_step >= cfg.max_steps:' in text


def test_openvla_action_heads_train_through_ddp_forward():
    """Parameterized training must call DDP, not methods on ``DDP.module``."""
    text = (
        REPO / "overlays/openvla/patches/openvla_oft_finetune_and_ddp.patch"
    ).read_text(encoding="utf-8")

    assert text.count("+    def forward(self, actions_hidden_states):") == 2
    assert "+            predicted_actions = action_head(actions_hidden_states)" in text
    assert "+            noise_pred = action_head(actions_hidden_states)" in text
    assert "+            predicted_actions = action_head.module.predict_action" not in text
    assert "+            noise_pred = action_head.module.predict_noise" not in text
    assert '"max_steps"' in text


def test_overfit_evaluation_separates_expert_gate_from_diagnostic_timeout():
    rollout = (REPO / "packages/panthera_sim/rollout.py").read_text(encoding="utf-8")
    evaluator = (
        REPO / "pipelines/ci/evaluate_numbered_overfit_checkpoints.sh"
    ).read_text(encoding="utf-8")
    parallel = (
        REPO / "pipelines/ci/rerun_extended_budget_overfit_evaluations.sh"
    ).read_text(encoding="utf-8")

    assert '"on_time_success"' in rollout
    assert '"delayed_success"' in rollout
    assert '"expert_action_budget": expert_budget' in rollout
    assert '--expert-action-budget "$expert_action_budget"' in evaluator
    assert '--max-actions "$max_actions"' in evaluator
    assert 'export CUDA_DEVICE_ORDER="$cuda_device_order"' in evaluator
    assert 'CI_OVERFIT_CUDA_DEVICE_ORDER:-PCI_BUS_ID' in evaluator
    assert evaluator.count(
        'CUDA_DEVICE_ORDER="$cuda_device_order" CUDA_VISIBLE_DEVICES="$gpu"'
    ) >= 3
    assert '--gpus "$gpu"' in evaluator
    assert "--gpus 0" not in evaluator
    assert 'CI_OVERFIT_EVAL_MAX_ACTIONS:-2074' in parallel
    assert 'CI_OVERFIT_EVAL_GPUS:-0,1,2,3' in parallel


def test_checkpoint_pair_queue_waits_for_atomic_checkpoint_completion():
    text = (
        REPO / "pipelines/ci/evaluate_numbered_overfit_checkpoint_pairs.sh"
    ).read_text(encoding="utf-8")

    assert "checkpoint_complete()" in text
    assert 'if ! checkpoint_complete "$checkpoint" "$step"' in text
    assert 'checkpoint ${step} 仍在写入' in text


def test_single_trajectory_gate_uses_separate_training_and_rollout_gpus():
    text = (REPO / "pipelines/ci/gate_single_trajectory_overfit.sh").read_text(
        encoding="utf-8"
    )
    assert 'train_gpus="${CI_OVERFIT_TRAIN_GPUS:-${CI_OVERFIT_GPUS:-0,1,2}}"' in text
    assert 'eval_gpu="${CI_OVERFIT_EVAL_GPU:-3}"' in text
    assert "CI_OVERFIT_ALLOW_SHARED_GPU" in text
