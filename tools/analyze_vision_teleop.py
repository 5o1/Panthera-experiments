#!/usr/bin/env python3
"""Convert a Panthera rosbag into machine-readable statistics, CSV and plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ros2_ws/src/panthera_vision_teleop"))

from panthera_vision_teleop.offline_analysis import SCALAR_TOPICS, VECTOR_TOPICS, recording_phase_intervals, summarize_recording, timing_summary


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a vision-teleoperation rosbag2 directory")
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, help="output directory; defaults to BAG/analysis")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _vector(message, topic: str):
    if topic in ("/teleop/human_pose", "/teleop/debug_target"):
        value = message.pose.position
    elif topic.startswith("/teleop/diagnostics/") or topic == "/teleop/depth_features":
        value = message.vector
    else:
        value = message
    return [float(value.x), float(value.y), float(value.z)]


def read_bag(path: Path):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise SystemExit("缺少 ROS 2 rosbag Python 依赖；请在 Panthera 系统 Python 环境运行。") from exc
    if not (path / "metadata.yaml").is_file():
        raise SystemExit(f"不是 rosbag2 目录（缺少 metadata.yaml）：{path}")
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    type_map = {item.name: item.type for item in reader.get_all_topics_and_types()}
    samples = {topic: [] for topic in (*VECTOR_TOPICS, *SCALAR_TOPICS)}
    timed_samples = {topic: [] for topic in samples}
    markers = []
    rows = []
    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic not in samples and topic != "/teleop/experiment_marker":
            continue
        if topic not in type_map:
            continue
        message = deserialize_message(data, get_message(type_map[topic]))
        timestamp_sec = timestamp_ns / 1e9
        if topic == "/teleop/experiment_marker":
            markers.append((timestamp_sec, str(message.data)))
            continue
        if topic in VECTOR_TOPICS:
            value = _vector(message, topic)
            row = (topic, timestamp_sec, *value)
        else:
            value = float(message.data)
            row = (topic, timestamp_sec, value, "", "")
        samples[topic].append(value)
        timed_samples[topic].append((timestamp_sec, value))
        rows.append(row)
    return {topic: values for topic, values in samples.items() if values}, rows, timed_samples, markers


def write_plot(samples, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topics = [topic for topic in (
        "/teleop/diagnostics/human_delta_camera",
        "/teleop/diagnostics/filtered_robot_offset",
        "/teleop/diagnostics/motion_limited_robot_offset",
        "/end_pose_euler",
    ) if topic in samples]
    if not topics:
        return
    figure, axes = plt.subplots(len(topics), 1, figsize=(11, 2.8 * len(topics)), squeeze=False)
    for axis, topic in zip(axes[:, 0], topics):
        values = samples[topic]
        axis.plot(values, linewidth=1.0)
        axis.set_title(topic)
        axis.set_ylabel("m")
        axis.legend(VECTOR_TOPICS[topic], loc="upper right")
        axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("sample index")
    figure.tight_layout()
    figure.savefig(output / "teleop_vectors.png", dpi=150)
    plt.close(figure)


def main() -> None:
    args = arguments()
    output = args.output or args.bag / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    samples, rows, timed_samples, markers = read_bag(args.bag)
    summary = summarize_recording(samples)
    summary["timing"] = {
        topic: timing_summary([timestamp for timestamp, _ in values])
        for topic, values in timed_samples.items()
        if values
    }
    phase_summaries = {}
    for name, (start, end) in recording_phase_intervals(markers).items():
        phase_samples = {}
        for topic, timed_values in timed_samples.items():
            selected = [value for timestamp, value in timed_values if start <= timestamp <= end]
            if selected:
                phase_samples[topic] = selected
        phase_summaries[name] = {
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            **summarize_recording(phase_samples),
        }
        phase_summaries[name]["timing"] = {
            topic: timing_summary([
                timestamp for timestamp, _ in timed_values if start <= timestamp <= end
            ])
            for topic, timed_values in timed_samples.items()
            if sum(1 for timestamp, _ in timed_values if start <= timestamp <= end) > 0
        }
    if phase_summaries:
        summary["phases"] = phase_summaries
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["topic", "time_sec", "x", "y", "z"])
        writer.writerows(rows)
    if not args.no_plot:
        write_plot(samples, output)
    print(f"分析完成：{output}")


if __name__ == "__main__":
    main()
