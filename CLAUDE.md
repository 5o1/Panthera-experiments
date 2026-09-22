# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two largely independent subsystems that share one Panthera robot arm as their target:

1. **`ros2_ws/` — real-time vision teleoperation.** A ROS 2 Humble package (`panthera_vision_teleop`) that drives the physical arm from monocular pose/gesture tracking. Runs locally (WSL + real hardware).
2. **`simulation/` + `tools/run_lab_*` — VLA (vision-language-action) training pipeline.** RoboTwin/SAPIEN simulation, RLinf, and OpenVLA-OFT fine-tuning for a single-arm "pick up cylinder, insert into socket" task. Runs on a remote GPU lab machine reached via `tools/lab_ssh.sh`; this repo holds the adapter/env code and orchestration scripts, not the training itself.

Read `AGENTS.md` first — it lists the required-reading doc order and states the current task boundaries (single Panthera arm, single gripper, 7-DoF contract: 6 joints + gripper). Dual-arm / dual-Piper / 14-DoF code and docs still present in the repo are historical exploration only and must not be reused for current training, eval, or real-hardware work.

## Subsystem 1: ROS 2 vision teleoperation

Data flow (see root `README.md` for the full diagram):

```
WSL camera → MediaPipe Pose/Gesture → WebSocket → vision_bridge_node → /teleop/*
  → teleop_mapper_node → continuous_cartesian_backend (50 Hz latest-only)
  → ros2_control (200 Hz) → Panthera SDK non-blocking command
```

Key design invariants (do not change without updating `AGENTS.md`/`docs/01_...`):
- **Latest-only, not FIFO.** The continuous path never queues motion targets; a stale/expired target causes hold-in-place + forced re-enable, never silent continuation.
- Real hardware only enables Y/Z by default; X (depth) and end-effector orientation are locked off (`enable_orientation: false`), gripper command publishing is a separate opt-in flag (`publish_gripper_commands`), because a relaxed hand can be misread as `Open_Palm`.
- Loss of tracking / low confidence / disconnect / stale source frames must latch to disabled — recovery of data must never auto-resume motion.
- Never start the official driver or publish real-robot commands without explicit user confirmation, a stated hardware confirmation word, and e-stop readiness.
- Motion limits are fixed, not tuned by trial on hardware: 0.6 rad/s cruise, 1.0 rad/s hardware cap, 2.0 rad/s² accel cap.

Module boundaries inside `ros2_ws/src/panthera_vision_teleop/panthera_vision_teleop/`:
- `vision_protocol.py`, `rotation_utils.py`, `teleop_logic.py` — pure Python, no ROS dependency, unit-testable core (protocol validation, arm/wrist frame math, calibration/mapping/gain/EMA/debounce logic).
- `vision_bridge_node.py` — isolates the WebSocket network thread from the ROS executor; disables output ~0.5s after disconnect.
- `teleop_mapper_node.py` — owns the dry-run vs. real-hardware boundary and the "at most one in-flight command" rule for the legacy `/pos_cmd` path.
- `kinematics.py`, `cartesian_trajectory_backend.py`, `continuous_cartesian_backend.py` — replaceable lightweight FK/Jacobian/IK backend (not MoveIt Servo).
- `debug_pose_publisher.py` / `manual_target_node.py` — camera/hardware-free synthetic inputs for testing.

Build and test:

```bash
source /opt/ros/humble/setup.bash
source ~/Panthera_HT_ROS2/install/setup.bash   # official driver underlay; must be sourced first
cd ~/panthera/ros2_ws
colcon build --symlink-install --packages-select panthera_vision_teleop
colcon test --packages-select panthera_vision_teleop
colcon test-result --verbose

cd ~/panthera
PYTHONPATH=ros2_ws/src/panthera_vision_teleop /usr/bin/python3 -m pytest -q ros2_ws/src/panthera_vision_teleop/test
# single test file:
PYTHONPATH=ros2_ws/src/panthera_vision_teleop /usr/bin/python3 -m pytest -q ros2_ws/src/panthera_vision_teleop/test/test_teleop_mapping.py
```

Full hardware-free software acceptance baseline (see `docs/05_software_acceptance.md`):

```bash
./tools/preflight_panthera.sh mock
./tools/test_process_cleanup.sh
./tools/test_vision_pipeline_mock.sh
./tools/test_blocking_scheduler_mock.sh
./tools/test_continuous_control_mock.sh
./tools/test_continuous_streaming_mock.sh
```

`test_continuous_*_mock.sh` hardcode `mock_components/GenericSystem` — they never load the Panthera SDK or touch a serial port. A software PASS does not validate camera noise, arm direction, serial timing, real braking, or physical e-stop; those require the graduated real-hardware checklist in `docs/04_real_hardware_experiment_checklist.md`.

End-to-end synthetic demo (no camera/robot needed): `./run_vision_teleop_demo.sh synthetic`. Camera dry-run: `./run_vision_teleop_demo.sh camera safe`. Real hardware (requires a typed confirmation phrase): `./run_vision_teleop_demo.sh robot safe`.

## Subsystem 2: Simulation / VLA pipeline

Restructured on 2026-09-19 (`docs/19_architecture_plan_2026-09-19.md`). The
governing rule is that **an artefact carries the configuration needed to
reproduce it**, and downstream reads the artefact rather than a config file
somewhere else. Every defect found in the 2026-09-18/19 investigation had the
same shape: a value that decided the result, in a place the consumer did not
know about, with nothing checking consistency.

```
packages/           importable, unit-tested Python
  panthera_sim/     config, assets, dataset access, executor, replay, rollout
  panthera_vla/     RLDS conversion, training glue, checkpoint contract, audits
overlays/           our files, destined for a generated runtime tree
  robotwin/{envs,task_config,description,assets/profiles,patches}
  rlinf/{config,evaluations,seeds,patches}
  openvla/patches
pipelines/          current entry points (assemble_runtime.py, ci/)
archive/            historical one-off entries and superseded implementations
tools/              Lab connection and local ROS 2 helpers
```

Four contracts hold it together:

- `packages/panthera_sim/config.py` is the **only** declaration of the 17
  `PANTHERA_*` simulation parameters, with defaults, ranges, and why each
  exists. `resolve()` materialises them; `snapshot()` is what a dataset stores.
- A **dataset** carries `dataset.json` (contract, sim config, assets, upstream
  commits) and `scenes.json` (seed → realized geometry). Open it with
  `panthera_sim.dataset.open_dataset`; a test fails the build if anything reads
  those files directly. `contract.required_action_budget` is measured from the
  recorded trajectories, so an evaluation cannot inherit a budget that is
  smaller than what the expert itself needs.
- A **checkpoint** carries `training.json` (action chunk, unnorm key, proprio
  and image contracts, robot platform, which dataset, hyperparameters). Open it
  with `panthera_vla.checkpoint.open_checkpoint`. Evaluating on a dataset the
  model was not trained on is refused unless `--allow-dataset-mismatch`.
- An **asset profile** (`overlays/robotwin/assets/profiles/`) owns an
  embodiment, its end effector's contact physics, and that end effector's
  behaviours — the gripper ratchet is declared there, enabled per context, and
  deliberately off during collection.

`packages/panthera_sim/executor.py` is the one executor. Expert replay and
policy rollout both drive it, because the replay's only purpose is to establish
the ceiling the rollout is measured against. Its defaults are the configuration
that produced the recorded baseline; the rejected alternatives are documented
next to them so they are not retried blind.

```bash
python -m pytest -q packages/ pipelines/   # 187 tests, no GPU or Lab needed

# Expert self-reproduction, the ceiling for any policy number
cd packages/panthera_sim && python3 replay.py \
  --dataset-root <dataset> --robotwin-root <RoboTwin> \
  --task-config panthera_phone_cylinder_socket_v2_pilot.yml \
  --sample 40 --workers 12 --gpus 0,1,2,3 --output <report>

# The two gates (docs/18 section 8); both are currently red
bash pipelines/ci/gate_expert_replay.sh
bash pipelines/ci/gate_single_trajectory_overfit.sh
```

Upstream (`RoboTwin`, `RLinf`, OpenVLA-OFT) is read-only and must stay
git-clean. `pipelines/assemble_runtime.py` builds a runtime tree from a pinned
checkout plus the overlay plus the patches, and refuses to run if the source has
local changes or was modified during assembly. Patches live beside the upstream
they modify, not in a single shared directory, and **one patch owns one upstream
file**: three patches organised by feature all changed `script/collect_data.py`,
each diffed against a tree where the others were already applied, so the second
one `git apply` reached was refused. Nothing had ever assembled from a clean
upstream, so that patch set had been unusable without anyone finding out.

On the Lab this is now in force rather than aspirational:

```
externals/RoboTwin   pinned at 6dde571, git-clean; the large asset directories
                     are symlinked, so it costs 12 MB instead of 16 GB
runtime/robotwin     generated, and verified byte-identical to the tree the
                     experiments actually ran on -- all six patched files, the
                     three task configs the overlay ships whole, and the three
                     task environments
```

Before this, the Lab's RoboTwin carried 424 untracked files and 9 modified ones,
deployed by copying straight into the upstream tree, and three of the nine
modifications were in no patch at all. Deploy by assembling; never write into
`externals/`.

The pin moved from `0008ae6` (2026-05-19) to `6dde571` 223 commits later on
2026-09-20, which was only doable because the changes were all in patches by
then: the conflicts could be measured on a throwaway clone instead of guessed
at. `docs/21` has what each patch became -- two were dropped because upstream
had adopted or removed them -- and the three substantive changes it forced,
the largest being that upstream now builds a `CuroboPlanner` unconditionally.
The mplib dispatch is restored by patch rather than switching planners, because
switching would void the expert-replay baseline every other number rests on.

Baselines, so a change that moves them is visible:

| Measurement | Value | Where |
|---|---|---|
| Expert replay, full set | 1249/1280 (upright 629/640, lying 620/640) | `docs/18` §11 |
| 16h policy on training scenes | 0/12, median 70 deg joint deviation | `docs/18` §9 |
| 16h policy, 18-seed eval | 0/18 after the harness was corrected | `docs/18` §7 |
| Single-trajectory overfit, recorded frames | 3.8 mrad median over the chunk, 20 frames spanning the episode | `docs/20` |
| Same model, sampled camera | 187.2 mrad median; never touches the cylinder | `docs/20` |
| Same model, recorded camera | 3.9 mrad median; grasps, lifts 10 cm, carries to 2.7 cm of the socket, never releases | `docs/20` |

**Every policy number above was measured through a randomly placed camera.**
`docs/20` is the correction: the dataset was collected with a pinned head
camera (`pose_mode: fixed`, hence its name `fixedcam`) and recorded that pose
per episode, but evaluation took its camera from whichever task config the
command line named, and the pilot config samples one (`target_xy_jitter_m:
0.06`). The camera sat 47.6 cm from the recorded viewpoint, looking from the
other side of the table. The same overfit model, same state, differs by a
median 187.2 mrad through the sampled camera and 3.9 mrad through the recorded
one, against 2.7 mrad for the recorded frame itself. `build_task` now requires
the recorded camera and verifies the placed pose, so the 0/18 and 0/12 rows are
void rather than pending. Read `docs/20` before treating any policy number as a
statement about the policy.

## Working across both subsystems

- `docs/NN_*.md` files are numbered chronologically and are the authoritative experiment/decision log — check the highest-numbered relevant doc (and `AGENTS.md`'s "当前边界" section) before assuming a config, task definition, or gate is still current; several documents explicitly supersede/veto earlier ones (e.g. docs 11–13 are rejected pilots, doc 15 is the accepted dataset).
- When a stage or hardware conclusion changes, update `docs/01_next_stage_vision_teleop_todo.md` (or the relevant design doc) and link it from `README.md` — don't leave conclusions only in chat history (stated maintenance rule in `AGENTS.md`).
- `bags/` (rosbag2 recordings) and Lab-side dataset/model directories are gitignored runtime data, not source.
