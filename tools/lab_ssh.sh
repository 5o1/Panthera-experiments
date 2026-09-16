#!/usr/bin/env bash

set -euo pipefail

windows_ssh="/mnt/c/Program Files/OpenSSH/ssh.exe"

if [[ ! -x "$windows_ssh" ]]; then
  echo "错误：未找到 Windows OpenSSH：${windows_ssh}" >&2
  exit 1
fi

exec "$windows_ssh" scalelab01 "$@"
