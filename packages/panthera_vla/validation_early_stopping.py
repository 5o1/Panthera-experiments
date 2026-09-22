#!/usr/bin/env python3
"""State machine for validation-loss early stopping.

The training process owns checkpoint I/O and distributed synchronization.  This
module only decides whether a validation result is a significant improvement
and whether patience has been exhausted, which keeps the policy independently
testable without importing PyTorch or OpenVLA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PassRateDecision:
    """Result of one closed-loop validation observation."""

    step: int
    pass_rate: float
    successes: int
    trials: int
    reached_target: bool
    consecutive_at_target: int
    should_stop: bool


class PassRateEarlyStopping:
    """Stop once the closed-loop pass rate reaches its target.

    Validation loss does not predict closed-loop success on this task: the
    historical model that grasped reliably scored a *worse* chunk L1 skill score
    and the same motion cosine as the model that scored 0/18.  Stopping on a
    loss plateau therefore stops on a quantity unrelated to the goal.  A pass
    rate stops on the goal itself, and on a single-trajectory overfit run it
    stops exactly when the run has proved what it set out to prove.

    ``minimum_step`` exists because the opposite mistake is also possible: an
    early validation that happens to succeed would end a run before the weights
    are worth keeping.
    """

    def __init__(
        self,
        *,
        target: float = 1.0,
        confirmations: int = 1,
        minimum_step: int = 0,
    ) -> None:
        if not 0.0 < target <= 1.0:
            raise ValueError("target must be in (0, 1]")
        if confirmations <= 0:
            raise ValueError("confirmations must be positive")
        if minimum_step < 0:
            raise ValueError("minimum_step must be non-negative")
        self.target = float(target)
        self.confirmations = int(confirmations)
        self.minimum_step = int(minimum_step)
        self.best_pass_rate: Optional[float] = None
        self.best_step: Optional[int] = None
        self.consecutive_at_target = 0
        self.validation_count = 0
        self.last_decision: Optional[PassRateDecision] = None

    def update(
        self, *, step: int, successes: int, trials: int
    ) -> PassRateDecision:
        """Consume one closed-loop validation result and decide whether to stop."""

        if step < 0:
            raise ValueError("step must be non-negative")
        if trials <= 0:
            raise ValueError("trials must be positive")
        if not 0 <= successes <= trials:
            raise ValueError("successes must be within [0, trials]")

        pass_rate = successes / trials
        reached = pass_rate >= self.target
        if reached:
            self.consecutive_at_target += 1
        else:
            self.consecutive_at_target = 0
        if self.best_pass_rate is None or pass_rate > self.best_pass_rate:
            self.best_pass_rate = pass_rate
            self.best_step = int(step)

        self.validation_count += 1
        decision = PassRateDecision(
            step=int(step),
            pass_rate=pass_rate,
            successes=int(successes),
            trials=int(trials),
            reached_target=reached,
            consecutive_at_target=self.consecutive_at_target,
            should_stop=(
                step >= self.minimum_step
                and self.consecutive_at_target >= self.confirmations
            ),
        )
        self.last_decision = decision
        return decision

    def as_dict(self, *, stop_reason: Optional[str] = None) -> dict:
        return {
            "criterion": "closed_loop_pass_rate",
            "target": self.target,
            "confirmations": self.confirmations,
            "minimum_step": self.minimum_step,
            "validation_count": self.validation_count,
            "best_pass_rate": self.best_pass_rate,
            "best_step": self.best_step,
            "consecutive_at_target": self.consecutive_at_target,
            "stop_reason": stop_reason,
            "last_validation": (
                asdict(self.last_decision) if self.last_decision is not None else None
            ),
        }

    def write_json(self, path: Path, *, stop_reason: Optional[str] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(self.as_dict(stop_reason=stop_reason), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


@dataclass(frozen=True)
class EarlyStoppingDecision:
    """Result of one validation observation."""

    step: int
    loss: float
    improvement: Optional[float]
    improved: bool
    bad_validation_count: int
    should_stop: bool


class ValidationEarlyStopping:
    """Stop after too many validation losses without a minimum improvement.

    The first finite loss establishes the baseline.  Later losses count as an
    improvement only when ``best_loss - loss >= min_delta``.  A smaller drop,
    an unchanged loss, or an increased loss consumes one patience slot.
    """

    def __init__(self, *, min_delta: float, patience: int) -> None:
        if not math.isfinite(min_delta) or min_delta <= 0.0:
            raise ValueError("min_delta must be finite and positive")
        if patience <= 0:
            raise ValueError("patience must be positive")
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.best_loss: Optional[float] = None
        self.best_step: Optional[int] = None
        self.bad_validation_count = 0
        self.validation_count = 0
        self.last_decision: Optional[EarlyStoppingDecision] = None

    def update(self, *, step: int, loss: float) -> EarlyStoppingDecision:
        """Consume one finite validation loss and return the stop decision."""

        loss = float(loss)
        if step < 0:
            raise ValueError("step must be non-negative")
        if not math.isfinite(loss):
            raise ValueError("validation loss must be finite")

        improvement = None if self.best_loss is None else self.best_loss - loss
        improved = self.best_loss is None or improvement >= self.min_delta
        if improved:
            self.best_loss = loss
            self.best_step = int(step)
            self.bad_validation_count = 0
        else:
            self.bad_validation_count += 1

        self.validation_count += 1
        decision = EarlyStoppingDecision(
            step=int(step),
            loss=loss,
            improvement=improvement,
            improved=improved,
            bad_validation_count=self.bad_validation_count,
            should_stop=self.bad_validation_count >= self.patience,
        )
        self.last_decision = decision
        return decision

    def as_dict(self, *, stop_reason: Optional[str] = None) -> dict:
        """Return a JSON-serializable training status snapshot."""

        return {
            "min_delta": self.min_delta,
            "patience": self.patience,
            "validation_count": self.validation_count,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "bad_validation_count": self.bad_validation_count,
            "stop_reason": stop_reason,
            "last_validation": (
                asdict(self.last_decision) if self.last_decision is not None else None
            ),
        }

    def write_json(self, path: Path, *, stop_reason: Optional[str] = None) -> None:
        """Atomically persist the current early-stopping state."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(self.as_dict(stop_reason=stop_reason), indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
