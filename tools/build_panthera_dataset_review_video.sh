#!/usr/bin/env bash

set -euo pipefail

dataset_root="/data/lyy/panthera-vla/data/place_vertical_cylinder_in_groove/panthera_phone_vertical_sft_v1"
report_root="/data/lyy/panthera-vla/reports/dataset-review"
stem="panthera_phone_vertical_sft_v1_episodes000-127_1024x768"
manifest="${report_root}/${stem}.concat.txt"
subtitles="${report_root}/${stem}.srt"
metadata="${report_root}/${stem}.source.json"
output="${report_root}/${stem}.mp4"
building="${output}.building"
summary="${report_root}/${stem}.summary.json"

mkdir -p "$report_root"

if [[ -e "$output" || -e "$building" ]]; then
  echo "错误：输出或半成品已经存在，拒绝覆盖：${output}" >&2
  exit 1
fi

python3 - "$dataset_root" "$manifest" "$subtitles" "$metadata" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


dataset_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
subtitle_path = Path(sys.argv[3])
metadata_path = Path(sys.argv[4])


def episode_id(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.mp4", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


videos = sorted((dataset_root / "video").glob("episode*.mp4"), key=episode_id)
ids = [episode_id(path) for path in videos]
if ids != list(range(128)):
    raise SystemExit(f"expected episode IDs 0..127, got {ids}")

entries = []
cursor = 0.0
manifest_lines = []
subtitle_blocks = []
for ordinal, path in enumerate(videos, start=1):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt:format=duration,size",
        "-of", "json", str(path),
    ], text=True))
    stream = probe["streams"][0]
    duration = float(probe["format"]["duration"])
    if (stream["width"], stream["height"], stream["r_frame_rate"]) != (320, 240, "30/1"):
        raise SystemExit(f"unexpected source format for {path}: {stream}")
    start = cursor
    end = cursor + duration
    episode = episode_id(path)
    manifest_lines.append(f"file '{path}'")
    subtitle_blocks.append(
        f"{ordinal}\n{srt_time(start)} --> {srt_time(end)}\n"
        f"episode {episode:03d}  ({ordinal}/128)\n"
    )
    entries.append({
        "episode": episode,
        "path": str(path),
        "duration_s": duration,
        "timeline_start_s": start,
        "timeline_end_s": end,
        "source_width": stream["width"],
        "source_height": stream["height"],
        "source_fps": stream["r_frame_rate"],
        "source_codec": stream["codec_name"],
        "source_bytes": int(probe["format"]["size"]),
    })
    cursor = end

manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
subtitle_path.write_text("\n".join(subtitle_blocks) + "\n", encoding="utf-8")
metadata_path.write_text(json.dumps({
    "dataset": "panthera_phone_vertical_sft_v1",
    "episode_count": len(entries),
    "expected_duration_s": cursor,
    "ordering": "numeric episode ID ascending",
    "sources": entries,
}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"episode_count": len(entries), "expected_duration_s": cursor}))
PY

ffmpeg -hide_banner -nostdin -y \
  -f concat -safe 0 -i "$manifest" \
  -vf "scale=1024:768:flags=lanczos,subtitles=${subtitles}:force_style='FontName=DejaVu Sans,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=1,Outline=2,Shadow=0,Alignment=7,MarginL=20,MarginV=20'" \
  -an -c:v h264_nvenc -gpu 1 -preset p5 -tune hq -rc vbr -cq 20 -b:v 0 \
  -pix_fmt yuv420p -movflags +faststart -f mp4 "$building"

python3 - "$building" "$metadata" "$summary" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

video = Path(sys.argv[1])
source_metadata = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
summary_path = Path(sys.argv[3])
probe = json.loads(subprocess.check_output([
    "ffprobe", "-v", "error", "-select_streams", "v:0",
    "-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt:format=duration,size",
    "-of", "json", str(video),
], text=True))
stream = probe["streams"][0]
duration = float(probe["format"]["duration"])
expected = float(source_metadata["expected_duration_s"])
if (stream["width"], stream["height"]) != (1024, 768):
    raise SystemExit(f"unexpected output dimensions: {stream}")
if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p":
    raise SystemExit(f"unexpected output codec: {stream}")
if abs(duration - expected) > 1.0:
    raise SystemExit(f"duration mismatch: output={duration}, expected={expected}")
digest = hashlib.sha256()
with video.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
result = {
    "status": "passed",
    "dataset": source_metadata["dataset"],
    "episode_count": source_metadata["episode_count"],
    "episode_order": "000..127",
    "output_width": stream["width"],
    "output_height": stream["height"],
    "output_fps": stream["r_frame_rate"],
    "output_codec": stream["codec_name"],
    "output_pixel_format": stream["pix_fmt"],
    "duration_s": duration,
    "size_bytes": int(probe["format"]["size"]),
    "sha256": digest.hexdigest(),
    "episode_labels_burned_in": True,
}
summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY

mv "$building" "$output"
echo "完成：${output}"
