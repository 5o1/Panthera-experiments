"""Frame-driven experiment phases shared by camera and robot data collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Optional


@dataclass(frozen=True)
class ExperimentPhase:
    name: str
    instruction: str
    warmup_valid_frames: int
    capture_valid_frames: int


class ExperimentSequence:
    """Advance phases by valid source frames, never by wall-clock sleeps."""

    def __init__(
        self,
        phases: list[ExperimentPhase],
        auto_advance_seconds: float = 0.0,
    ) -> None:
        if not phases:
            raise ValueError("experiment needs at least one phase")
        if len({phase.name for phase in phases}) != len(phases):
            raise ValueError("experiment phase names must be unique")
        for phase in phases:
            if not phase.name or len(phase.name) > 48 or not phase.name.replace("_", "").isalnum():
                raise ValueError("phase names must be short alphanumeric identifiers")
            if phase.warmup_valid_frames < 0 or phase.capture_valid_frames < 1:
                raise ValueError("phase frame counts are invalid")
        if not math.isfinite(auto_advance_seconds) or auto_advance_seconds < 0.0:
            raise ValueError("auto advance seconds must be finite and non-negative")
        self.phases = phases
        self.auto_advance_seconds = auto_advance_seconds
        self.index = 0
        self.state = "WAITING"
        self.remaining = 0
        self._intermission_deadline: Optional[float] = None

    @classmethod
    def from_json(cls, path: Path | str) -> "ExperimentSequence":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("phases"), list):
            raise ValueError("experiment plan must contain a phases array")
        phases = []
        for value in data["phases"]:
            if not isinstance(value, dict):
                raise ValueError("every phase must be an object")
            phases.append(ExperimentPhase(
                name=str(value["name"]),
                instruction=str(value["instruction"]),
                warmup_valid_frames=int(value.get("warmup_valid_frames", 30)),
                capture_valid_frames=int(value["capture_valid_frames"]),
            ))
        return cls(phases, float(data.get("auto_advance_seconds", 0.0)))

    @property
    def current(self) -> Optional[ExperimentPhase]:
        return None if self.index >= len(self.phases) else self.phases[self.index]

    @property
    def phase_is_armed(self) -> bool:
        """Whether a real-robot source may be enabled for the current phase."""
        return self.state in ("WARMUP", "RECORDING")

    def arm_current(self) -> str:
        phase = self.current
        if phase is None:
            return "ALL_COMPLETE"
        if self.state not in ("WAITING", "WAITING_NEXT"):
            return "ALREADY_RUNNING"
        if phase.warmup_valid_frames:
            self.state = "WARMUP"
            self.remaining = phase.warmup_valid_frames
            return "WARMUP_STARTED"
        self.state = "RECORDING"
        self.remaining = phase.capture_valid_frames
        return "RECORDING_STARTED"

    def update(self, pose_valid: bool, now_s: Optional[float] = None) -> Optional[str]:
        now = time.monotonic() if now_s is None else now_s
        if self.state == "INTERMISSION":
            assert self._intermission_deadline is not None
            seconds_left = self._intermission_deadline - now
            self.remaining = max(0, math.ceil(seconds_left))
            if seconds_left > 0.0:
                return None
            phase = self.current
            assert phase is not None
            self._intermission_deadline = None
            if phase.warmup_valid_frames:
                self.state = "WARMUP"
                self.remaining = phase.warmup_valid_frames
            else:
                self.state = "RECORDING"
                self.remaining = phase.capture_valid_frames
            return f"NEXT_PHASE_STARTED:{phase.name}"
        if self.state not in ("WARMUP", "RECORDING") or not pose_valid:
            return None
        self.remaining -= 1
        if self.remaining > 0:
            return None
        phase = self.current
        assert phase is not None
        if self.state == "WARMUP":
            self.state = "RECORDING"
            self.remaining = phase.capture_valid_frames
            return "RECORDING_STARTED"
        completed_name = phase.name
        self.index += 1
        if self.current is None:
            self.state = "COMPLETE"
        elif self.auto_advance_seconds:
            self.state = "INTERMISSION"
            self._intermission_deadline = now + self.auto_advance_seconds
            self.remaining = math.ceil(self.auto_advance_seconds)
        else:
            self.state = "WAITING_NEXT"
        return f"PHASE_COMPLETE:{completed_name}"

    def marker(self) -> str:
        phase = self.current
        name = "none" if phase is None else phase.name
        return f"{name}:{self.state}:{self.remaining}"
