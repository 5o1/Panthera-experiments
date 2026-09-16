#!/usr/bin/env bash

set -euo pipefail

workspace="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
rlinf_root="${workspace}/RLinf"
robotwin_root="${workspace}/RoboTwin"
conda_root="${PANTHERA_CONDA_ROOT:-/data/lyy/tools/miniforge3}"
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rlinf_cuda_arch_patch="${script_root}/patches/rlinf_robotwin_cuda_arch.patch"
rlinf_curobo_patch="${script_root}/patches/rlinf_robotwin_curobo_v1.patch"
rlinf_curobo_metadata_patch="${script_root}/patches/rlinf_robotwin_curobo_metadata.patch"
state_root="${workspace}/.bootstrap-state"
cache_root="${workspace}/cache"
tmp_root="${workspace}/tmp"
nas_root="${PANTHERA_NAS_ROOT:-/mnt/hulab/pro6000/data}"

rlinf_commit="a3816b596478dcd8a5c69a6ec1468c9519f77b5b"
robotwin_commit="0008ae6800df9f75fc8de7098bacb01735fd8fd2"

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

mkdir -p "$state_root" "$cache_root/huggingface" "$cache_root/uv" "$tmp_root"
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

apply_checkout_patch() {
  local repository=$1
  local patch_file=$2
  local description=$3

  if [[ ! -f "$patch_file" ]]; then
    echo "错误：缺少补丁：${patch_file}" >&2
    exit 1
  fi
  if git -C "$repository" apply --reverse --check "$patch_file" \
      >/dev/null 2>&1; then
    echo "${description}已应用，跳过。"
  elif git -C "$repository" apply --check "$patch_file"; then
    git -C "$repository" apply "$patch_file"
  else
    echo "错误：${description}与固定源码版本不兼容。" >&2
    exit 1
  fi
}

echo "[1/5] 验证固定源码版本"
verify_checkout "$rlinf_root" "$rlinf_commit"
verify_checkout "$robotwin_root" "$robotwin_commit"
apply_checkout_patch "$rlinf_root" "$rlinf_cuda_arch_patch" \
  "RLinf CUDA 架构补丁"
apply_checkout_patch "$rlinf_root" "$rlinf_curobo_patch" \
  "RLinf cuRobo v1 固定补丁"
apply_checkout_patch "$rlinf_root" "$rlinf_curobo_metadata_patch" \
  "RLinf cuRobo v1 wheel 元数据补丁"
touch "${state_root}/source.ok"

echo "[2/5] 导入 NAS 独有的轻量 walkthrough"
if [[ ! -f "${state_root}/walkthrough.ok" ]]; then
  source_notebooks="${nas_root}/RLinf/experiment_notebooks"
  if [[ ! -d "$source_notebooks" ]]; then
    echo "错误：NAS walkthrough 目录不可用：${source_notebooks}" >&2
    exit 1
  fi
  mkdir -p "${rlinf_root}/experiment_notebooks"
  for filename in \
    place_empty_cup_walkthrough.py \
    robotwin_rlinf_guide.md \
    run_place_empty_cup_walkthrough.sh; do
    rsync -a "${source_notebooks}/${filename}" \
      "${rlinf_root}/experiment_notebooks/${filename}"
  done

  if [[ ! -e "${rlinf_root}/RoboTwin" ]]; then
    ln -s ../RoboTwin "${rlinf_root}/RoboTwin"
  elif [[ $(readlink -f "${rlinf_root}/RoboTwin") != "$robotwin_root" ]]; then
    echo "错误：${rlinf_root}/RoboTwin 已存在且目标不正确。" >&2
    exit 1
  fi
  touch "${state_root}/walkthrough.ok"
else
  echo "walkthrough 已导入，跳过。"
fi

echo "[3/5] 按 RLinf 官方原生方式构建 OpenVLA-OFT + RoboTwin 环境"
if [[ ! -f "${state_root}/environment.ok" ]]; then
  if [[ ! -x "${conda_root}/bin/conda" ]]; then
    echo "错误：缺少 Miniforge：${conda_root}" >&2
    exit 1
  fi
  if [[ ! -x "${rlinf_root}/.venv/bin/python" ]]; then
    "${conda_root}/bin/conda" create --yes --prefix "${rlinf_root}/.venv" \
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
  if [[ ! -e "${rlinf_root}/.venv/bin/activate" ]]; then
    touch "${rlinf_root}/.venv/bin/activate"
  fi
  export VIRTUAL_ENV="${rlinf_root}/.venv"
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

if [[ ! -e "${rlinf_root}/.venv-robotwin" ]]; then
  ln -s .venv "${rlinf_root}/.venv-robotwin"
elif [[ $(readlink -f "${rlinf_root}/.venv-robotwin") != "${rlinf_root}/.venv" ]]; then
  echo "错误：${rlinf_root}/.venv-robotwin 已存在且目标不正确。" >&2
  exit 1
fi

echo "[4/5] 从官方 Hugging Face 源下载并配置 RoboTwin assets"
if [[ ! -f "${state_root}/assets.ok" ]]; then
  cd "$robotwin_root"
  bash script/_download_assets.sh
  touch "${state_root}/assets.ok"
else
  echo "RoboTwin assets 已下载，跳过。"
fi

echo "[5/5] 验证 Python、GPU、RLinf 与 RoboTwin"
cd "$rlinf_root"
PYTHONPATH="${robotwin_root}:${rlinf_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  python - <<'PY'
import importlib
from pathlib import Path

import torch

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

robotwin_root = Path("/data/lyy/panthera-vla/RoboTwin")
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
