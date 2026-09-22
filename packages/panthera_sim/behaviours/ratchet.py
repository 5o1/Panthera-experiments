"""Hold what the gripper has closed on, whatever the commanded opening does.

This is a property of the end effector, not of whichever script is driving it:
an evaluation, a real deployment and a diagnostic replay all want the same rule,
and it was previously a private class inside one audit script.  The asset
profile declares it and its constants; callers ask the profile for it.

A larger value is a wider opening.  Arm jitter is absorbed by the controller and
the task tolerates centimetres of arm error, but a single frame of extra opening
drops the object and nothing recovers from that, so the two dimensions do not
deserve the same treatment.  Release stays possible because an intended release
is large and sustained while jitter is small and transient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GripperRatchet:
    """Suppress opening while holding; let closing through unchanged."""

    margin: float = 0.15
    hold_steps: int = 3
    contact_threshold: int = 2
    holding: bool = field(default=False, init=False)
    held_value: Optional[float] = field(default=None, init=False)
    release_votes: int = field(default=0, init=False)
    engaged_at: Optional[int] = field(default=None, init=False)
    suppressed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.margin <= 1.0:
            raise ValueError("margin must be in (0, 1]")
        if self.hold_steps < 1:
            raise ValueError("hold_steps must be positive")
        if self.contact_threshold < 1:
            raise ValueError("contact_threshold must be positive")

    def reset(self) -> None:
        self.holding = False
        self.held_value = None
        self.release_votes = 0
        self.engaged_at = None
        self.suppressed = 0

    def __call__(self, commanded: float, contacts: int, step: int) -> float:
        if not self.holding:
            if contacts >= self.contact_threshold:
                self.holding = True
                self.held_value = commanded
                self.engaged_at = step
            return commanded
        if commanded > self.held_value + self.margin:
            self.release_votes += 1
            if self.release_votes >= self.hold_steps:
                self.holding = False
                return commanded
        else:
            self.release_votes = 0
        if commanded < self.held_value:
            self.held_value = commanded  # the ratchet only ever tightens
        elif commanded > self.held_value:
            self.suppressed += 1
        return self.held_value

    def telemetry(self) -> dict:
        return {
            "engaged_at": self.engaged_at,
            "suppressed_openings": self.suppressed,
            "still_holding": self.holding,
        }
