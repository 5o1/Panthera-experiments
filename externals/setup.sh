#!/usr/bin/env bash
# Fetch the pinned upstream checkouts.
#
# They are read-only inputs: the runtime tree used at run time is generated from
# them by pipelines/assemble_runtime.py, which refuses to build from a checkout
# that has local changes.  That refusal is the point -- our task environments,
# instructions, generated configs and audit scripts previously lived inside
# these trees as 415 untracked and 9 modified files with no local commits, so
# "what have we changed" had no answer.
#
# Nothing here is small: about 28 GB between the two. They are deliberately not
# vendored; the commit of each is recorded in every dataset the collector wrote.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pinned="${root}/pinned.json"

read_field() {
  python3 -c "import json,sys;print(json.load(open('${pinned}'))['upstreams']['$1']['$2'])"
}

for name in robotwin rlinf; do
  url="$(read_field "$name" url)"
  commit="$(read_field "$name" commit)"
  target="${root}/$(read_field "$name" directory)"
  if [[ -d "${target}/.git" ]]; then
    current="$(git -C "$target" rev-parse HEAD)"
    if [[ "$current" == "$commit" ]]; then
      echo "  ${name} 已是 ${commit:0:12}"
      continue
    fi
    echo "错误：${target} 在 ${current:0:12}，期望 ${commit:0:12}；请手动核对后再切换。" >&2
    exit 1
  fi
  echo "  克隆 ${name} <- ${url}"
  branch="$(read_field "$name" branch)"
  git clone --filter=blob:none --branch "$branch" "$url" "$target"
  git -C "$target" checkout --detach "$commit"
done

echo "完成。上游为只读输入，请勿在其中编辑；运行时目录由 pipelines/assemble_runtime.py 生成。"
