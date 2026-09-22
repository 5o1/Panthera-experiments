#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
rlinf_source="${workspace}/externals/RLinf"
robotwin_source="${workspace}/externals/RoboTwin"
rlinf_root="${workspace}/runtime/rlinf"
robotwin_root="${workspace}/runtime/robotwin"
venv_root="${workspace}/envs/rlinf"
conda_root="${PANTHERA_CONDA_ROOT:-/data/lyy/tools/miniforge3}"
state_root="${workspace}/state/bootstrap-state"
cache_root="${workspace}/cache"
tmp_root="${workspace}/tmp"

rlinf_commit="a3816b596478dcd8a5c69a6ec1468c9519f77b5b"
robotwin_commit="6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755"

if [[ $(id -u) -eq 0 ]]; then
  echo "错误：本脚本必须使用普通用户运行，禁止使用 root。" >&2
  exit 1
fi

for command_name in git flock rsync; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "错误：缺少命令：${command_name}" >&2
    exit 1
  fi
done

mkdir -p "$state_root" "$cache_root/huggingface" "$cache_root/uv" "$tmp_root" \
  "$(dirname "$venv_root")"
exec 9>"${state_root}/bootstrap.lock"
if ! flock -n 9; then
  echo "错误：另一个 Panthera VLA 构建进程正在运行。" >&2
  exit 1
fi

export HF_HOME="${cache_root}/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_CACHE_DIR="${cache_root}/uv"
export XDG_CACHE_HOME="$cache_root"
export TMPDIR="$tmp_root"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${conda_root}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ROBOTWIN_PATH="$robotwin_root"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# The Lab uses RTX 4090 GPUs (Ada, compute capability 8.9).  CUDA 13.2 no
# longer accepts legacy compute_70, which PyTorch3D otherwise adds when it
# builds for every architecture known to PyTorch.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-16}"
# PyTorch3D v0.7.9 predates CUDA 13's device-symbol visibility change.
# These are the upstream CUDA 13 compatibility flags now used by PyTorch3D.
if [[ -z ${NVCC_FLAGS:-} ]]; then
  export NVCC_FLAGS="--device-entity-has-hidden-visibility=false -static-global-template-stub=false"
fi

verify_checkout() {
  local repository=$1
  local expected_commit=$2
  local actual_commit

  if [[ ! -d "${repository}/.git" ]]; then
    echo "错误：源码仓库不存在：${repository}" >&2
    exit 1
  fi
  actual_commit=$(git -C "$repository" rev-parse HEAD)
  if [[ "$actual_commit" != "$expected_commit" ]]; then
    echo "错误：${repository} 版本不匹配：${actual_commit}" >&2
    echo "预期：${expected_commit}" >&2
    exit 1
  fi
}

echo "[1/5] 验证固定源码版本"
verify_checkout "$rlinf_source" "$rlinf_commit"
verify_checkout "$robotwin_source" "$robotwin_commit"
python3 "${workspace}/pipelines/assemble_runtime.py" \
  --upstream rlinf --source "$rlinf_source" --runtime "$rlinf_root"
python3 "${workspace}/pipelines/assemble_runtime.py" \
  --upstream robotwin --source "$robotwin_source" --runtime "$robotwin_root"
if [[ -e "${rlinf_root}/.venv" && ! -L "${rlinf_root}/.venv" ]]; then
  echo "错误：${rlinf_root}/.venv 不是受管符号链接，拒绝覆盖。" >&2
  exit 1
fi
ln -sfn "$venv_root" "${rlinf_root}/.venv"
touch "${state_root}/source.ok"

echo "[2/5] 按 RLinf 官方原生方式构建 OpenVLA-OFT + RoboTwin 环境"
if [[ ! -f "${state_root}/environment.ok" || ! -x "${venv_root}/bin/python" ]]; then
  if [[ ! -x "${conda_root}/bin/conda" ]]; then
    echo "错误：缺少 Miniforge：${conda_root}" >&2
    exit 1
  fi
  if [[ ! -x "${venv_root}/bin/python" ]]; then
    "${conda_root}/bin/conda" create --yes --prefix "$venv_root" \
      python=3.11.14 pip uv
  fi
  # Use the Conda prefix as RLinf's official .venv target. The upstream
  # installer detects and reuses its Python 3.11 interpreter, then performs
  # the locked uv installation inside that prefix.
  # shellcheck disable=SC1091
  source "${conda_root}/etc/profile.d/conda.sh"
  conda activate "${rlinf_root}/.venv"
  # RLinf's installer recognizes an existing environment only when this
  # venv-style marker exists. Conda activation already owns PATH and Python;
  # the empty compatibility marker prevents uv from trying to recreate the
  # populated prefix, while VIRTUAL_ENV makes `uv sync --active` target it.
  if [[ ! -e "${venv_root}/bin/activate" ]]; then
    touch "${venv_root}/bin/activate"
  fi
  export VIRTUAL_ENV="$venv_root"
  cd "$rlinf_root"
  bash requirements/install.sh embodied \
    --model openvla-oft \
    --env robotwin \
    --no-root \
    --no-flash-attn \
    --no-apex \
    --install-rlinf
  touch "${state_root}/environment.ok"
else
  echo "Python 环境已构建，跳过。"
fi

# shellcheck disable=SC1091
source "${conda_root}/etc/profile.d/conda.sh"
conda activate "${rlinf_root}/.venv"

resolved_venv=$(readlink -f "${rlinf_root}/.venv")
if [[ ! -e "${rlinf_root}/.venv-robotwin" ]]; then
  ln -s .venv "${rlinf_root}/.venv-robotwin"
elif [[ $(readlink -f "${rlinf_root}/.venv-robotwin") != "$resolved_venv" ]]; then
  echo "错误：${rlinf_root}/.venv-robotwin 已存在且目标不正确。" >&2
  exit 1
fi

echo "[3/5] 应用受管 OpenVLA-OFT 补丁"
openvla_root=$(python - <<'PY'
from importlib.metadata import distribution

# Package import executes prismatic.constants before the Panthera patch exists
# and prints its LIBERO fallback to stdout, corrupting command substitution.
# Distribution metadata locates site-packages without executing package code.
print(distribution("openvla-oft").locate_file(""))
PY
)
openvla_git_ceiling=$(dirname "$openvla_root")
patch_state="${state_root}/openvla-patches"
mkdir -p "$patch_state"
shopt -s nullglob
patches=("${workspace}"/overlays/openvla/patches/*.patch)
previous_patches=("${patch_state}"/*.patch)
patch_set_changed=0
for patch in "${patches[@]}"; do
  name=$(basename "$patch")
  previous="${patch_state}/${name}"
  if [[ ! -s "$previous" ]] || ! cmp -s "$previous" "$patch"; then
    patch_set_changed=1
  fi
done
for previous in "${previous_patches[@]}"; do
  name=$(basename "$previous")
  if [[ ! -s "${workspace}/overlays/openvla/patches/${name}" ]]; then
    patch_set_changed=1
  fi
done

# A feature patch may absorb a previously separate patch.  In that case the
# new combined diff cannot be applied on top of either old one.  Remove the
# complete recorded set in reverse order, then install the complete current
# set; never attempt an in-place mixture of generations.
if [[ "$patch_set_changed" == "1" ]]; then
  for ((index=${#previous_patches[@]} - 1; index >= 0; index--)); do
    previous=${previous_patches[$index]}
    if ! GIT_CEILING_DIRECTORIES="$openvla_git_ceiling" \
      git -C "$openvla_root" apply --reverse --check "$previous"; then
      echo "错误：无法干净撤销旧 OpenVLA 补丁：$(basename "$previous")" >&2
      echo "请重建 ${venv_root}，不要继续使用混合补丁环境。" >&2
      exit 1
    fi
    GIT_CEILING_DIRECTORIES="$openvla_git_ceiling" \
      git -C "$openvla_root" apply --reverse "$previous"
    rm -f "$previous"
  done
fi

for patch in "${patches[@]}"; do
  name=$(basename "$patch")
  previous="${patch_state}/${name}"
  if GIT_CEILING_DIRECTORIES="$openvla_git_ceiling" \
    git -C "$openvla_root" apply --check "$patch"; then
    GIT_CEILING_DIRECTORIES="$openvla_git_ceiling" \
      git -C "$openvla_root" apply "$patch"
  elif GIT_CEILING_DIRECTORIES="$openvla_git_ceiling" \
    git -C "$openvla_root" apply --reverse --check "$patch"; then
    echo "  ${name} 已应用，跳过。"
  else
    echo "错误：OpenVLA 补丁无法应用：${name}" >&2
    exit 1
  fi
  cp "$patch" "$previous"
done
touch "${state_root}/openvla-patches.ok"

echo "[4/5] 从官方 Hugging Face 源下载并配置 RoboTwin assets"
if [[ ! -f "${state_root}/assets.ok" ]]; then
  cd "$robotwin_root"
  bash scripts/_download_assets.sh
  touch "${state_root}/assets.ok"
else
  echo "RoboTwin assets 已下载，跳过。"
fi

echo "[5/5] 验证 Python、GPU、RLinf 与 RoboTwin"
cd "$rlinf_root"
ROBOT_PLATFORM=PANTHERA \
PYTHONPATH="${robotwin_root}:${rlinf_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  python - <<'PY'
import importlib
from pathlib import Path

import torch
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM

modules = (
    "rlinf",
    "sapien",
    "mplib",
    "pytorch3d",
    "curobo.types.math",
    "curobo.wrap.reacher.motion_gen",
)
for module_name in modules:
    importlib.import_module(module_name)

robotwin_root = Path("/data/lyy/panthera-vla/runtime/robotwin")
required_paths = (
    robotwin_root / "envs" / "place_empty_cup.py",
    robotwin_root / "assets" / "objects",
    robotwin_root / "assets" / "embodiments",
)
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise SystemExit(f"RoboTwin 资源不完整：{missing}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch 未检测到 CUDA")
if torch.cuda.device_count() != 4:
    raise SystemExit(f"预期 4 张 GPU，实际 {torch.cuda.device_count()} 张")
if (ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM) != (7, 25, 7):
    raise SystemExit(
        "Panthera OpenVLA 契约错误："
        f"dim={ACTION_DIM}, chunk={NUM_ACTIONS_CHUNK}, proprio={PROPRIO_DIM}"
    )

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"gpu_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
print("RLinf/RoboTwin 基础环境验证通过")
PY

touch "${state_root}/verified.ok"
date --iso-8601=seconds >"${state_root}/completed-at.txt"
echo "构建完成：${workspace}"
