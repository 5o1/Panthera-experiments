#!/usr/bin/env bash
#
# Mirror the Lab's deployed code into a local directory.
#
# WSL cannot reach the Lab at all: the route goes through Windows, which does
# not forward to 172.17.20.31, so ping is 100% loss and port 22 times out.
# Only the Windows OpenSSH client has a path, which is why `lab_ssh.sh` shells
# out to it -- and why a plain `sshfs` from here cannot work.
#
# A real mount needs two things that are one command away but not installable
# without a password:
#
#   sudo apt install -y sshfs                          # in WSL
#   sudo mkdir -p /mnt/gpu_node && sudo chown $(id -u):$(id -g) /mnt/gpu_node
#   ./tools/lab_mirror.sh tunnel              # a background Windows-side forward
#   ./tools/lab_mirror.sh mount
#
# The mount point is /mnt/gpu_node, outside the repository. Mounting the Lab
# inside the working tree once let an `rm` aimed at a local directory delete
# the Lab through it; outside, the same mistake cannot reach.
#
# Until then this keeps a local copy in sync, which covers reading and editing
# without a round trip per file. Only the deployed code is mirrored; the Lab's
# ~700 GB of datasets, runs and caches are not, and each of those already has
# its own source.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote="scalelab01:/data/lyy/panthera-vla"
local_dir="${PANTHERA_GPU_NODE:-/mnt/gpu_node}"
ssh_shim="${repo}/tools/lib/windows_ssh.sh"
# The tracked set, matching the Lab's own .gitignore.
paths=(bin packages pipelines overlays docs patches CLAUDE.md .gitignore)

ensure_local_dir() {
  [[ -d "$local_dir" ]] && return
  cat >&2 <<MSG
错误：挂载点 ${local_dir} 不存在。它故意放在仓库之外——针对工作目录的 rm
永远够不到 Lab。创建一次即可：

  sudo mkdir -p ${local_dir} && sudo chown "$(id -u):$(id -g)" ${local_dir}
MSG
  exit 1
}

usage() {
  cat >&2 <<USAGE
用法: $(basename "$0") <pull|push|diff|tunnel|proxy|mount|umount>

  pull    Lab → ${local_dir}
  push    ${local_dir} → Lab   （先看 diff，再推）
  diff    只报告差异，不传输
  tunnel  在后台起一条 Windows 侧端口转发，供 sshfs 使用
  proxy   把 WSL 这边能用的代理反向转发到 Lab（Lab 自己的代理上游已断）
  mount   sshfs 挂载（需要先 tunnel，且已安装 sshfs）
  umount  卸载
USAGE
  exit 2
}

[[ $# -ge 1 ]] || usage
mkdir -p "$(dirname "$ssh_shim")"
cat > "$ssh_shim" <<'SHIM'
#!/usr/bin/env bash
exec "/mnt/c/Program Files/OpenSSH/ssh.exe" "$@"
SHIM
chmod +x "$ssh_shim"

case "$1" in
  pull)
    ensure_local_dir
    for p in "${paths[@]}"; do
      rsync -e "$ssh_shim" -a --delete --info=stats1 \
        "${remote}/${p}" "${local_dir}/" 2>&1 | grep -E "^(sent|total)" || true
    done
    echo "已拉取到 ${local_dir}"
    ;;
  push)
    for p in "${paths[@]}"; do
      [[ -e "${local_dir}/${p}" ]] || continue
      rsync -e "$ssh_shim" -a --info=stats1 \
        "${local_dir}/${p}" "${remote}/" 2>&1 | grep -E "^(sent|total)" || true
    done
    echo "已推送到 Lab；记得在 Lab 上提交（那边的工作区受 git 管理）"
    ;;
  diff)
    for p in "${paths[@]}"; do
      out=$(rsync -e "$ssh_shim" -ain --delete "${remote}/${p}" "${local_dir}/" 2>/dev/null || true)
      [[ -n "$out" ]] && { echo "--- ${p}"; echo "$out"; }
    done
    echo "（空=一致）"
    ;;
  tunnel)
    "/mnt/c/Program Files/OpenSSH/ssh.exe" -N \
      -L "192.168.16.1:2222:172.17.20.31:22" scalelab01 &
    echo "转发已起：WSL 可经 192.168.16.1:2222 到达 Lab（pid $!）"
    ;;
  mount)
    command -v sshfs >/dev/null || { echo "错误：未安装 sshfs（sudo apt install -y sshfs）" >&2; exit 1; }
    ensure_local_dir
    if mountpoint -q "$local_dir"; then echo "已经挂载在 ${local_dir}"; exit 0; fi
    sshfs -p 2222 "lyy@192.168.16.1:/data/lyy/panthera-vla" "$local_dir" \
      -o IdentityFile="$HOME/.ssh/lyy@scalelab",reconnect,ServerAliveInterval=15
    echo "已挂载到 ${local_dir}"
    ;;
  proxy)
    # The Lab has no working route to PyPI or GitHub. Its own proxy (xray,
    # 127.0.0.1:17890) accepts the connection and its Japan upstream returns
    # nothing -- 0 bytes after 20s, while the xray log shows the request being
    # accepted. So forward the proxy that does work, the one WSL itself uses.
    #
    # The port must be one nothing on the Lab already holds: 17890 is xray's,
    # and binding over it fails in a way the ssh client still reports as
    # "remote forward success", which sends every request into the dead proxy.
    port="${PANTHERA_LAB_PROXY_PORT:-18899}"
    upstream="${PANTHERA_WSL_PROXY:-192.168.16.1:7897}"
    "/mnt/c/Program Files/OpenSSH/ssh.exe" -N \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -R "${port}:${upstream}" scalelab01 &
    echo "代理已转发：Lab 上 http://127.0.0.1:${port} -> ${upstream}（pid $!）"
    echo "在 Lab 上用：export https_proxy=http://127.0.0.1:${port} http_proxy=\$https_proxy"
    ;;
  umount)
    fusermount3 -u "$local_dir" 2>/dev/null || fusermount -u "$local_dir"
    echo "已卸载"
    ;;
  *) usage ;;
esac
