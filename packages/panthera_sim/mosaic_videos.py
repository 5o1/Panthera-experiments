#!/usr/bin/env python3
"""Place checkpoint rollout videos on one synchronized comparison canvas."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _step(path: Path) -> int:
    match = re.search(r"step-(\d+)", path.stem)
    return int(match.group(1)) if match else 10**18


def _checkpoint_label(path: Path) -> str:
    label = f"checkpoint {_step(path)}"
    report_path = path.with_suffix(".json")
    if not report_path.is_file():
        return label + " | offline val L1 unavailable"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        loss = report["cases"][0]["offline_validation"]["normalized_l1"]
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return label + " | offline val L1 unavailable"
    horizon = report.get("execution_horizon")
    ensemble = report.get("temporal_ensemble")
    if horizon == 1 and ensemble == 0.0:
        mode = "h1 equal sliding mean"
    elif horizon is not None:
        mode = f"h{horizon} raw"
    else:
        mode = None
    if mode is None:
        return label + f" | offline val L1 {float(loss):.6f}"
    return label + f" | {mode} | offline val L1 {float(loss):.6f}"


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    cell = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    cell[y : y + resized_h, x : x + resized_w] = resized
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    paths = sorted((path.resolve() for path in args.input), key=_step)
    if not paths or any(not path.is_file() for path in paths):
        raise SystemExit("every --input must be an existing video")
    columns = math.ceil(math.sqrt(len(paths)))
    rows = math.ceil(len(paths) / columns)
    cell_w, cell_h = args.width // columns, args.height // rows
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if any(not capture.isOpened() for capture in captures):
        raise SystemExit("could not open every input video")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "bgr24", "-video_size", f"{args.width}x{args.height}",
            "-framerate", str(args.fps), "-i", "-", "-pix_fmt", "yuv420p",
            "-vcodec", "libx264", "-crf", "21", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    last: list[np.ndarray | None] = [None] * len(captures)
    frames_written = 0
    try:
        while True:
            advanced = False
            canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            for index, (path, capture) in enumerate(zip(paths, captures)):
                ok, frame = capture.read()
                if ok:
                    last[index] = frame
                    advanced = True
                if last[index] is None:
                    continue
                cell = _letterbox(last[index], cell_w, cell_h)
                label = _checkpoint_label(path)
                cv2.rectangle(cell, (0, 0), (cell_w, 34), (0, 0, 0), -1)
                cv2.putText(
                    cell, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                row, column = divmod(index, columns)
                y, x = row * cell_h, column * cell_w
                canvas[y : y + cell_h, x : x + cell_w] = cell
            if not advanced:
                break
            assert encoder.stdin is not None
            encoder.stdin.write(canvas.tobytes())
            frames_written += 1
    finally:
        for capture in captures:
            capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg failed while encoding the comparison video")
    if frames_written == 0:
        raise SystemExit("input videos contained no frames")
    print(
        f"{args.output}: {len(paths)} videos, {columns}x{rows} grid, "
        f"{frames_written} frames at {args.fps:g} fps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
