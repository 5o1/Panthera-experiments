#!/usr/bin/env python3
"""Monocular MediaPipe sender for WSL/Linux (also works on Windows).

This process owns only the camera and human observations. It never imports ROS
and can never command the robot directly. Space toggles enable, R requests one
calibration packet, and Esc exits.
"""

from __future__ import annotations

import argparse
import asyncio
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import websockets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSE_MODEL = ROOT / "docs/references/models/pose_landmarker_full.task"
DEFAULT_GESTURE_MODEL = ROOT / "docs/references/models/gesture_recognizer.task"
PACKAGE_SOURCE = ROOT / "ros2_ws/src/panthera_vision_teleop"
HUD_FONT_CANDIDATES = (
    Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)

# Share the safety rule with the ROS-side protocol tests without importing ROS.
sys.path.insert(0, str(PACKAGE_SOURCE))
from panthera_vision_teleop.vision_protocol import latch_enable_on_pose_loss
from panthera_vision_teleop.experiment_sequence import ExperimentSequence


@lru_cache(maxsize=8)
def _hud_font(size: int) -> ImageFont.FreeTypeFont:
    for path in HUD_FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise RuntimeError("no Chinese HUD font found; expected Microsoft YaHei or Noto Sans CJK")


def _draw_unicode_hud(
    image: Any,
    lines: list[tuple[tuple[int, int], str, int, tuple[int, int, int]]],
) -> Any:
    """Draw readable Chinese HUD lines once per frame using the available CJK font."""
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for position, value, size, color in lines:
        font = _hud_font(size)
        bounds = draw.textbbox(position, value, font=font)
        draw.rectangle((bounds[0] - 5, bounds[1] - 3, bounds[2] + 5, bounds[3] + 3), fill=(0, 0, 0))
        draw.text(position, value, font=font, fill=color)
    return np.asarray(canvas).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send monocular arm/gesture observations to ROS")
    parser.add_argument("--server", default="ws://127.0.0.1:8765")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--pose-score-min", type=float, default=0.6)
    parser.add_argument("--fourcc", default="MJPG", help="V4L2 pixel format; MJPG avoids usbipd YUYV green frames")
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--gesture-model", type=Path, default=DEFAULT_GESTURE_MODEL)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--enabled", action="store_true", help="start enabled (camera mode defaults disabled)")
    parser.add_argument("--synthetic", action="store_true", help="exercise WebSocket without a camera/models")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds; 0 means until Esc/Ctrl-C")
    parser.add_argument("--experiment-plan", type=Path, help="frame-driven JSON data-collection plan")
    parser.add_argument(
        "--experiment-require-enabled",
        action="store_true",
        help="count experiment frames only while teleoperation is enabled",
    )
    return parser.parse_args()


class LatestConnection:
    """Reconnect without retaining stale observations."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.websocket = None
        self.next_attempt = 0.0
        self.reported_connected = False

    async def send(self, packet: dict[str, Any]) -> Optional[dict[str, Any]]:
        now = time.monotonic()
        if self.websocket is None and now >= self.next_attempt:
            try:
                self.websocket = await asyncio.wait_for(websockets.connect(self.uri, max_size=65536), timeout=2.0)
                print(f"Connected to {self.uri}")
                self.reported_connected = True
            except Exception as exc:
                self.next_attempt = now + 1.0
                print(f"Waiting for bridge at {self.uri}: {type(exc).__name__}")
                return None
        if self.websocket is None:
            return None
        try:
            await self.websocket.send(json.dumps(packet, separators=(",", ":"), allow_nan=False))
            try:
                raw_reply = await asyncio.wait_for(self.websocket.recv(), timeout=0.2)
                reply = json.loads(raw_reply)
                return reply if isinstance(reply, dict) else {}
            except asyncio.TimeoutError:
                return {}
        except Exception as exc:
            print(f"Bridge disconnected: {type(exc).__name__}")
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
            self.next_attempt = now + 0.5
            return None

    async def close(self) -> None:
        if self.websocket is not None:
            await self.websocket.close()


def _timestamp_ms(previous: int) -> int:
    return max(previous + 1, int(time.monotonic() * 1000.0))


def _blank_packet(seq: int, timestamp_ms: int, enabled: bool, recalibrate: bool) -> dict[str, Any]:
    return {
        "v": 1, "seq": seq, "source_time_ms": timestamp_ms,
        "enabled": enabled, "recalibrate": recalibrate,
        "pose_valid": False, "hand_valid": False,
        "shoulder": [0.0, 0.0, 0.0], "elbow": [0.0, 0.0, 0.0], "wrist": [0.0, 0.0, 0.0],
        "pose_score": 0.0, "gesture": "Unknown", "gesture_score": 0.0,
        "depth_world_m": 0.0, "image_reach_ratio": 0.0,
        "image_arm_scale": 0.0, "pinch_ratio": -1.0,
        "depth_clutch": False, "orientation_valid": False,
        "experiment_marker": "",
    }


def _distance2(left: Any, right: Any) -> float:
    return math.hypot(float(left.x) - float(right.x), float(left.y) - float(right.y))


def _arm_plane_is_valid(shoulder: Any, elbow: Any, wrist: Any, epsilon: float = 0.05) -> bool:
    upper = (elbow.x - shoulder.x, elbow.y - shoulder.y, elbow.z - shoulder.z)
    forearm = (wrist.x - elbow.x, wrist.y - elbow.y, wrist.z - elbow.z)
    cross = (
        upper[1] * forearm[2] - upper[2] * forearm[1],
        upper[2] * forearm[0] - upper[0] * forearm[2],
        upper[0] * forearm[1] - upper[1] * forearm[0],
    )
    upper_norm = math.sqrt(sum(value * value for value in upper))
    forearm_norm = math.sqrt(sum(value * value for value in forearm))
    cross_norm = math.sqrt(sum(value * value for value in cross))
    return upper_norm > 1e-6 and forearm_norm > 1e-6 and cross_norm / (upper_norm * forearm_norm) >= epsilon


def _extract_pose(result: Any, packet: dict[str, Any]) -> Optional[list[Any]]:
    if not result.pose_world_landmarks:
        return None
    landmarks = result.pose_world_landmarks[0]
    selected = [landmarks[index] for index in (12, 14, 16)]
    packet["shoulder"] = [float(v) for v in (selected[0].x, selected[0].y, selected[0].z)]
    packet["elbow"] = [float(v) for v in (selected[1].x, selected[1].y, selected[1].z)]
    packet["wrist"] = [float(v) for v in (selected[2].x, selected[2].y, selected[2].z)]
    packet["pose_score"] = float(min(getattr(point, "visibility", 0.0) for point in selected))
    packet["pose_valid"] = all(math.isfinite(value) for name in ("shoulder", "elbow", "wrist") for value in packet[name])
    packet["depth_world_m"] = packet["wrist"][2] - packet["shoulder"][2]
    packet["orientation_valid"] = _arm_plane_is_valid(*selected)
    normalized = result.pose_landmarks[0] if result.pose_landmarks else None
    if normalized is not None:
        shoulder_width = _distance2(normalized[11], normalized[12])
        if shoulder_width > 1e-6:
            packet["image_reach_ratio"] = _distance2(normalized[12], normalized[16]) / shoulder_width
            packet["image_arm_scale"] = (
                _distance2(normalized[12], normalized[14])
                + _distance2(normalized[14], normalized[16])
            ) / shoulder_width
    return normalized


def _extract_right_gesture(result: Any, packet: dict[str, Any]) -> None:
    for index, handedness in enumerate(result.handedness):
        if not handedness or handedness[0].category_name.lower() != "right":
            continue
        if index >= len(result.gestures) or not result.gestures[index]:
            continue
        category = result.gestures[index][0]
        packet["hand_valid"] = True
        packet["gesture"] = category.category_name or "Unknown"
        packet["gesture_score"] = float(category.score)
        if index < len(result.hand_landmarks):
            hand = result.hand_landmarks[index]
            palm_width = _distance2(hand[5], hand[17])
            if palm_width > 1e-6:
                packet["pinch_ratio"] = _distance2(hand[4], hand[8]) / palm_width
        return


def _draw_preview(
    frame: Any,
    pose_landmarks: Optional[list[Any]],
    packet: dict[str, Any],
    fps: float,
    telemetry: dict[str, Any],
    experiment: Optional[ExperimentSequence],
    pose_score_min: float,
    experiment_require_enabled: bool,
) -> None:
    height, width = frame.shape[:2]
    if pose_landmarks is not None:
        points = []
        for index in (12, 14, 16):
            point = pose_landmarks[index]
            pixel = int(point.x * width), int(point.y * height)
            points.append(pixel)
            cv2.circle(frame, pixel, 7, (0, 255, 0), -1)
        cv2.line(frame, points[0], points[1], (0, 220, 255), 3)
        cv2.line(frame, points[1], points[2], (0, 220, 255), 3)

    # Mirror only what the operator sees.  The MediaPipe input and transmitted
    # coordinates above stay in the camera's original, unmirrored convention.
    # Drawing the HUD after the flip keeps its text readable.
    display = cv2.flip(frame, 1)
    unicode_hud: list[tuple[tuple[int, int], str, int, tuple[int, int, int]]] = []
    camera_only = experiment is not None and not experiment_require_enabled
    status = "CAMERA ONLY" if camera_only else ("ENABLED" if packet["enabled"] else "DISABLED")
    status_color = (120, 255, 255) if camera_only else ((0, 255, 0) if packet["enabled"] else (0, 0, 255))
    cv2.putText(display, f"{status}  {packet['gesture']} {packet['gesture_score']:.2f}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    clutch = "ON" if packet["depth_clutch"] else "OFF"
    pose_ready = bool(packet["pose_valid"]) and float(packet["pose_score"]) >= pose_score_min
    pose_label = "OK" if pose_ready else f"TOO LOW (<{pose_score_min:.2f})"
    cv2.putText(display, f"pose {packet['pose_score']:.2f} {pose_label}  FPS {fps:.1f}  depth clutch {clutch}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if pose_ready else (0, 0, 255), 2)
    human_delta = telemetry.get("human_delta", [0.0, 0.0, 0.0])
    target_delta = telemetry.get("target_delta", [0.0, 0.0, 0.0])
    flags = "CLIPPED" if telemetry.get("clipped") else ""
    if telemetry.get("motion_limited"):
        flags = (flags + " RATE-LIMITED").strip()
    if telemetry.get("orientation_held"):
        flags = (flags + " ORIENTATION-HELD").strip()
    cv2.putText(display, f"{telemetry.get('state', 'UNKNOWN')}  {flags}", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    cv2.putText(display, f"dHuman {human_delta[0]:+.3f} {human_delta[1]:+.3f} {human_delta[2]:+.3f}", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, f"dTarget {target_delta[0]:+.3f} {target_delta[1]:+.3f} {target_delta[2]:+.3f}", (18, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if experiment is not None:
        phase = experiment.current
        phase_name = "complete" if phase is None else phase.name
        phase_number = min(experiment.index + 1, len(experiment.phases))
        remaining_unit = "s" if experiment.state == "INTERMISSION" else "frames"
        cv2.putText(display, f"CASE {phase_number}/{len(experiment.phases)} {phase_name} {experiment.state} {remaining_unit}={experiment.remaining}", (18, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 255), 1)
        recorder = "READY" if telemetry.get("recording_ready", False) else "MISSING"
        cv2.putText(display, f"RECORDER {recorder}", (18, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 255) if recorder == "READY" else (0, 0, 255), 1)
        if phase is not None:
            action_prefix = {
                "WAITING": "准备好后按 Space 开始" if experiment_require_enabled else "准备好后按回车",
                "WAITING_NEXT": "准备好后按 Space 开始" if experiment_require_enabled else "准备好后按回车",
                "INTERMISSION": f"下一动作，{experiment.remaining} 秒后开始",
                "WARMUP": "按 Space 恢复跟随" if experiment_require_enabled and not packet["enabled"] else "准备",
                "RECORDING": "现在执行",
            }.get(experiment.state, experiment.state)
            action_color = (0, 255, 0) if experiment.state == "RECORDING" else (0, 220, 255)
            unicode_hud.append(((18, 198), f"{action_prefix}：{phase.instruction}", 20, action_color))
    if camera_only:
        unicode_hud.append(((18, height - 38), "镜像预览｜回车开始首阶段｜Esc 退出", 17, (255, 255, 255)))
    elif experiment is not None:
        unicode_hud.append(((18, height - 38), "每阶段只按一次 Space：装载并启用｜Esc 退出", 17, (255, 255, 255)))
    else:
        unicode_hud.append(((18, height - 38), "Space 启用｜R 重新标定｜X 深度离合｜Esc 退出", 17, (255, 255, 255)))
    display = _draw_unicode_hud(display, unicode_hud)
    cv2.imshow("Panthera monocular teleop (display mirrored only)", display)


async def run_synthetic(args: argparse.Namespace) -> None:
    connection = LatestConnection(args.server)
    start, seq, timestamp = time.monotonic(), 0, 0
    period = 1.0 / args.fps
    print("Synthetic vision enabled: no camera or model is being used")
    try:
        while not args.duration or time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            timestamp = _timestamp_ms(timestamp)
            phase = 2.0 * math.pi * (loop_start - start) / 8.0
            packet = _blank_packet(seq, timestamp, True, seq == 0)
            packet.update({
                "pose_valid": True, "hand_valid": True,
                "shoulder": [0.0, 0.0, 0.0], "elbow": [0.12, 0.18, 0.02],
                "wrist": [0.34 + 0.08*math.sin(phase), 0.12 + 0.05*math.cos(phase), -0.10],
                "pose_score": 0.99,
                "gesture": "Open_Palm" if int((loop_start-start)//3) % 2 == 0 else "Closed_Fist",
                "gesture_score": 0.95,
                "depth_world_m": -0.1,
                "image_reach_ratio": 2.0 + 0.1*math.sin(phase),
                "image_arm_scale": 2.5,
                "pinch_ratio": 0.8,
                "orientation_valid": True,
            })
            await connection.send(packet)
            seq += 1
            await asyncio.sleep(max(0.0, period - (time.monotonic() - loop_start)))
    finally:
        await connection.close()


async def run_camera(args: argparse.Namespace) -> None:
    for model in (args.pose_model, args.gesture_model):
        if not model.is_file():
            raise FileNotFoundError(f"model not found: {model}")
    capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if len(args.fourcc) != 4:
        raise ValueError("--fourcc must contain exactly four characters")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera index {args.camera}; check /dev/video* and usbipd attach")

    # UVC devices often return uninitialized frames immediately after opening.
    # Warm up before constructing MediaPipe inputs and report the negotiated mode.
    warm_frame = None
    for _ in range(10):
        ok, candidate = capture.read()
        if ok:
            warm_frame = candidate
    actual_fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(chr((actual_fourcc_value >> (8 * i)) & 0xFF) for i in range(4))
    print(
        "Camera opened: "
        f"/dev/video{args.camera} {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} {actual_fourcc}"
    )
    if warm_frame is None:
        capture.release()
        raise RuntimeError("camera opened but returned no frames")
    channel_mean = warm_frame.mean(axis=(0, 1))
    channel_std = warm_frame.std(axis=(0, 1))
    if float(channel_mean.max()) < 15.0 and float(channel_std.max()) < 5.0:
        print(
            "WARNING: camera image is nearly black; open the physical privacy shutter "
            "and ensure the lens has light before enabling teleoperation"
        )

    vision = mp.tasks.vision
    pose_options = vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(args.pose_model)), running_mode=vision.RunningMode.VIDEO, num_poses=1)
    gesture_options = vision.GestureRecognizerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(args.gesture_model)), running_mode=vision.RunningMode.VIDEO, num_hands=2)
    connection = LatestConnection(args.server)
    experiment = ExperimentSequence.from_json(args.experiment_plan) if args.experiment_plan else None
    if experiment is not None and experiment.current is not None:
        print(f"Data case ready: {experiment.current.name}: {experiment.current.instruction}")
        start_key = "Space" if args.experiment_require_enabled else "Enter"
        print(f"Pose stable 后按 {start_key}；预热和采集均按有效帧推进，不使用定时 sleep。")
    enabled, recalibrate, depth_clutch, seq, timestamp = args.enabled, False, False, 0, 0
    started, last_frame = time.monotonic(), time.monotonic()
    telemetry: dict[str, Any] = {}
    try:
        with vision.PoseLandmarker.create_from_options(pose_options) as pose_model, vision.GestureRecognizer.create_from_options(gesture_options) as gesture_model:
            while not args.duration or time.monotonic() - started < args.duration:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("camera stopped returning frames")
                timestamp = _timestamp_ms(timestamp)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pose_result = pose_model.detect_for_video(image, timestamp)
                gesture_result = gesture_model.recognize_for_video(image, timestamp)
                packet = _blank_packet(seq, timestamp, enabled, recalibrate)
                packet["depth_clutch"] = depth_clutch
                landmarks = _extract_pose(pose_result, packet)
                _extract_right_gesture(gesture_result, packet)
                recording_ready = bool(telemetry.get("recording_ready", False))
                if experiment is not None and telemetry and not recording_ready and enabled:
                    enabled = False
                    depth_clutch = False
                    recalibrate = False
                    print("Recorder subscriber lost: teleoperation latched DISABLED and phase count paused.")
                camera_experiment_ready = (
                    experiment is not None
                    and not args.experiment_require_enabled
                    and experiment.phase_is_armed
                    and bool(packet["pose_valid"])
                    and float(packet["pose_score"]) >= args.pose_score_min
                    and recording_ready
                )
                if camera_experiment_ready and not enabled:
                    # Camera cases cannot move hardware.  Keep dry-run mapping
                    # diagnostics alive across phase boundaries and pose loss
                    # without asking the operator for extra key presses.
                    enabled = True
                    recalibrate = True
                    packet["enabled"] = True
                    packet["recalibrate"] = True
                # Label the frame with the phase state in which it was acquired.
                # Updating first would incorrectly count the final warmup frame as
                # recording and omit the final recording frame from offline slices.
                if experiment is not None:
                    packet["experiment_marker"] = experiment.marker()
                experiment_event = experiment.update(
                    bool(packet["pose_valid"])
                    and float(packet["pose_score"]) >= args.pose_score_min
                    and recording_ready
                    and (enabled or not args.experiment_require_enabled)
                ) if experiment is not None else None
                if experiment_event is not None:
                    print(f"\a{experiment_event}")
                    if (
                        experiment_event.startswith("NEXT_PHASE_STARTED")
                        and not args.experiment_require_enabled
                    ):
                        enabled = True
                        recalibrate = True
                    if experiment_event.startswith("PHASE_COMPLETE"):
                        # A long case keeps the driver and bag alive, but each
                        # motion phase ends at a deterministic safe boundary.
                        # The operator must explicitly re-enable the next phase.
                        enabled = False
                        depth_clutch = False
                        recalibrate = False
                        print("Phase boundary: teleoperation latched DISABLED.")
                    if experiment_event.startswith("PHASE_COMPLETE") and experiment.current is not None:
                        print(f"Next phase: {experiment.current.name}: {experiment.current.instruction}")
                        if experiment.state == "INTERMISSION":
                            print(
                                f"将在 {experiment.remaining} 秒倒计时后自动开始；"
                                "无需再次按 Enter。"
                            )
                        elif args.experiment_require_enabled:
                            print("准备好后按一次 Space；程序会装载阶段并启用，等待期间 rosbag 保持连续录制。")
                        else:
                            print("准备好后按 Enter；等待期间 rosbag 保持连续录制。")
                enabled, tracking_lost = latch_enable_on_pose_loss(
                    enabled,
                    bool(packet["pose_valid"]),
                    float(packet["pose_score"]),
                    args.pose_score_min,
                )
                packet["enabled"] = enabled
                if tracking_lost:
                    recalibrate = False
                    depth_clutch = False
                    packet["recalibrate"] = False
                    packet["depth_clutch"] = False
                    print(
                        "Pose tracking lost: teleoperation latched DISABLED; "
                        "restore a clear view, then press Space to re-enable"
                    )
                telemetry_reply = await connection.send(packet)
                delivered = telemetry_reply is not None
                if telemetry_reply:
                    telemetry = telemetry_reply
                if enabled and not delivered:
                    enabled = False
                    depth_clutch = False
                    packet["enabled"] = False
                    print(
                        "Vision bridge unavailable: teleoperation latched DISABLED; "
                        "press Space after reconnection to re-enable"
                    )
                recalibrate, seq = False, seq + 1

                now = time.monotonic()
                fps = 1.0 / max(now - last_frame, 1e-6)
                last_frame = now
                key = -1
                if not args.no_preview:
                    _draw_preview(
                        frame,
                        landmarks,
                        packet,
                        fps,
                        telemetry,
                        experiment,
                        args.pose_score_min,
                        args.experiment_require_enabled,
                    )
                    key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord(" "):
                    if experiment is not None and args.experiment_require_enabled:
                        source_ready = (
                            bool(packet["pose_valid"])
                            and float(packet["pose_score"]) >= args.pose_score_min
                            and bool(telemetry.get("recording_ready", False))
                        )
                        if not enabled and not source_ready:
                            enabled = False
                            depth_clutch = False
                            print("START REFUSED：需要 pose OK 且 RECORDER READY；机械臂保持 DISABLED。")
                        elif enabled:
                            enabled = False
                            depth_clutch = False
                            print("Teleoperation DISABLED")
                        elif experiment.phase_is_armed:
                            enabled = True
                            print("Teleoperation ENABLED")
                        else:
                            event = experiment.arm_current()
                            if event in ("WARMUP_STARTED", "RECORDING_STARTED"):
                                enabled = True
                                print(f"\a{event}: {experiment.marker()}")
                                print("Teleoperation ENABLED：阶段已原子装载并启用。")
                            else:
                                enabled = False
                                print(f"START REFUSED: {event}")
                    else:
                        enabled = not enabled
                        if not enabled:
                            depth_clutch = False
                        print("Teleoperation ENABLED" if enabled else "Teleoperation DISABLED")
                elif key in (ord("r"), ord("R")):
                    recalibrate = True
                    print("Recalibration requested")
                elif key in (ord("x"), ord("X")):
                    depth_clutch = not depth_clutch
                    print("Depth clutch ON" if depth_clutch else "Depth clutch OFF")
                elif key in (10, 13) and experiment is not None:
                    if args.experiment_require_enabled:
                        print("真机 case 每阶段只需按 Space；Enter 不执行操作。")
                    elif not telemetry.get("recording_ready", False):
                        print("Recorder is not ready; phase remains waiting.")
                    else:
                        event = experiment.arm_current()
                        if event in ("WARMUP_STARTED", "RECORDING_STARTED"):
                            enabled = True
                            recalibrate = True
                        print(f"\a{event}: {experiment.marker()}")
                if experiment is not None and experiment.state == "COMPLETE":
                    print("All experiment phases complete; exiting through the normal cleanup path.")
                    break
                await asyncio.sleep(0)
    finally:
        capture.release()
        cv2.destroyAllWindows()
        await connection.close()


def main() -> None:
    args = parse_args()
    if args.fps <= 0.0:
        raise SystemExit("--fps must be positive")
    if not 0.0 <= args.pose_score_min <= 1.0:
        raise SystemExit("--pose-score-min must be in [0, 1]")
    try:
        asyncio.run(run_synthetic(args) if args.synthetic else run_camera(args))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f"vision_sender: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
