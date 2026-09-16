#!/usr/bin/env bash

vla_root="${PANTHERA_VLA_ROOT:-/data/lyy/panthera-vla}"
rlinf_root="${vla_root}/RLinf"
robotwin_root="${vla_root}/RoboTwin"
conda_root="${PANTHERA_CONDA_ROOT:-/data/lyy/tools/miniforge3}"

if [[ ! -x "${conda_root}/bin/conda" ]]; then
  echo "错误：未找到 Miniforge：${conda_root}" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -x "${rlinf_root}/.venv/bin/python" ]]; then
  echo "错误：Lab VLA 环境尚未构建：${rlinf_root}/.venv" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${conda_root}/etc/profile.d/conda.sh"
conda activate "${rlinf_root}/.venv"
export PANTHERA_VLA_ROOT="$vla_root"
export PANTHERA_CONDA_ROOT="$conda_root"
export RLINF_ROOT="$rlinf_root"
export ROBOTWIN_PATH="$robotwin_root"
export ASSETS_PATH="$robotwin_root"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HOME="${vla_root}/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_CACHE_DIR="${vla_root}/cache/uv"
export XDG_CACHE_HOME="${vla_root}/cache"
export TMPDIR="${vla_root}/tmp"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-16}"
if [[ -z ${NVCC_FLAGS:-} ]]; then
  export NVCC_FLAGS="--device-entity-has-hidden-visibility=false -static-global-template-stub=false"
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${robotwin_root}:${rlinf_root}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Panthera Lab VLA 环境已激活：${vla_root}"
