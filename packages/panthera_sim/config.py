"""Single declaration of every simulation parameter the task reads.

Seventeen ``PANTHERA_*`` environment variables were being read by
``os.environ.get`` calls scattered across three environment modules, each with
its default and range written beside it and nowhere listed together.  Counting
them was unreliable -- a first attempt missed the multi-line calls, including
the contact-solver settings that were under investigation at the time -- and
none of the resolved values reached the dataset, so a dataset could not say what
physics it had been collected under.

This module is the only place a parameter is declared.  ``resolve`` turns the
process environment into a fully materialised :class:`SimConfig`, and that object
is what a dataset snapshots and a downstream run is checked against.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


class SimConfigError(ValueError):
    """A parameter is missing, unparsable, or outside its declared range."""


@dataclass(frozen=True)
class Parameter:
    """One declared simulation parameter.

    ``why`` is not decoration: several of these values are the difference
    between an episode succeeding and failing, and the reason was previously
    recoverable only from a commit message or a document.
    """

    name: str
    env: Optional[str]
    default: Any
    kind: Callable[[str], Any]
    why: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[tuple] = None
    exclusive_minimum: bool = False

    def parse(self, raw: Optional[str]) -> Any:
        if raw is None:
            return self.default
        try:
            value = self.kind(raw)
        except (TypeError, ValueError) as error:
            raise SimConfigError(
                f"{self.env or self.name}={raw!r} is not a valid "
                f"{self.kind.__name__}"
            ) from error
        return value

    def validate(self, value: Any) -> None:
        source = self.env or self.name
        if self.choices is not None:
            if value not in self.choices:
                raise SimConfigError(
                    f"{source}={value!r} must be one of {self.choices}"
                )
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        if not math.isfinite(float(value)):
            raise SimConfigError(f"{source}={value!r} must be finite")
        if self.minimum is not None:
            too_small = (
                value <= self.minimum if self.exclusive_minimum else value < self.minimum
            )
            if too_small:
                boundary = ">" if self.exclusive_minimum else ">="
                raise SimConfigError(
                    f"{source}={value} must be {boundary} {self.minimum}"
                )
        if self.maximum is not None and value > self.maximum:
            raise SimConfigError(f"{source}={value} must be <= {self.maximum}")


def _boolean(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{raw!r} is not a boolean")


# Geometry is measured from the built embodiment and the task assets; it is
# declared here so the dataset can record what it was collected against, and is
# deliberately not overridable -- changing it changes the task, not a setting.
CYLINDER_RADIUS_M = 0.0275
CYLINDER_HALF_HEIGHT_M = 0.06
TABLE_HEIGHT_M = 0.74
SPAWN_CLEARANCE_M = 0.002

PARAMETERS: tuple[Parameter, ...] = (
    Parameter(
        "cylinder_linear_damping", "PANTHERA_V2_CYLINDER_LINEAR_DAMPING", 1.0, float,
        "Applied when the gripper opens. Too low and a seated cylinder never "
        "comes to rest, which fails the two at-rest success criteria.",
        minimum=0.0, maximum=10.0,
    ),
    Parameter(
        "cylinder_angular_damping", "PANTHERA_V2_CYLINDER_ANGULAR_DAMPING", 2.0, float,
        "Same release moment as the linear term.",
        minimum=0.0, maximum=10.0,
    ),
    Parameter(
        "solver_position_iterations", "PANTHERA_V2_SOLVER_POSITION_ITERATIONS", 20, int,
        "Contact accuracy at release. At 20 the solver leaves residual "
        "penetration that it then resolves by injecting velocity every step, so "
        "11 of 1280 expert replays oscillate forever in the socket. Raising it "
        "to 64 stops that but cost 13 lying episodes on the full set, so the "
        "default stays 20 and the defect is open.",
        minimum=1, maximum=255,
    ),
    Parameter(
        "solver_velocity_iterations", "PANTHERA_V2_SOLVER_VELOCITY_ITERATIONS", 4, int,
        "Velocity-pass counterpart of the position iterations.",
        minimum=1, maximum=255,
    ),
    Parameter(
        "max_depenetration_velocity_mps",
        "PANTHERA_V2_MAX_DEPENETRATION_VELOCITY_MPS", 0.2, float,
        "Caps how fast the solver may push overlapping bodies apart, and so "
        "how much energy an unconverged contact can inject per step.",
        minimum=0.0, maximum=10.0, exclusive_minimum=True,
    ),
    Parameter(
        "lying_grasp_axis_offset_m", "PANTHERA_V2_LYING_GRASP_AXIS_OFFSET_M", 0.044,
        float,
        "How far along its own axis a lying cylinder is grasped from the end.",
        minimum=0.0, maximum=CYLINDER_HALF_HEIGHT_M - 0.015, exclusive_minimum=True,
    ),
    Parameter(
        "direct_release_bottom_clearance_m",
        "PANTHERA_V2_DIRECT_RELEASE_BOTTOM_CLEARANCE_M", -0.014, float,
        "Cylinder bottom relative to the socket top at release; negative means "
        "inserted. Releasing above this is what leaves an episode toppling.",
        minimum=-0.035, maximum=0.020,
    ),
    Parameter(
        "reorientation_release_z_m", "PANTHERA_V2_REORIENTATION_RELEASE_Z_M", 0.86,
        float,
        "Height the cylinder is released at during a lying reorientation.",
        minimum=0.81, maximum=0.905,
    ),
    Parameter(
        "forced_lying_angle_bin", "PANTHERA_V2_FORCED_LYING_ANGLE_BIN", None, int,
        "Collection-side sharding control. Overrides the seed's own angle "
        "sector, and the override reaches the dataset only through "
        "scene_info.json -- which is why a seed does not identify a scene.",
        minimum=0, maximum=7,
    ),
    Parameter(
        "grasp_axis_sign", "PANTHERA_V2_GRASP_AXIS_SIGN", None, float,
        "Forces which end of a lying cylinder is grasped; planning picks it "
        "when unset.",
        choices=(-1.0, 1.0),
    ),
    Parameter(
        "reorientation_radial_mode", "PANTHERA_V2_REORIENTATION_RADIAL_MODE", None, str,
        "Forces the reorientation station; planning picks it when unset.",
        choices=("base", "side"),
    ),
    Parameter(
        "side_grasp_yaw_deg", "PANTHERA_V2_SIDE_GRASP_YAW_DEG", None, float,
        "Yaw of a side grasp approach.",
        minimum=-180.0, maximum=180.0,
    ),
    Parameter(
        "terminal_target_assist", "PANTHERA_TERMINAL_TARGET_ASSIST", False, _boolean,
        "Hand-engineered terminal skill. Any run with this on is not measuring "
        "the policy alone and must say so.",
    ),
    Parameter(
        "terminal_target_assist_trigger", "PANTHERA_TERMINAL_TARGET_ASSIST_TRIGGER",
        "height", str,
        "What triggers the terminal assist when it is enabled.",
        choices=("height", "release"),
    ),
    Parameter(
        "terminal_insertion_assist_m", "PANTHERA_TERMINAL_INSERTION_ASSIST_M", 0.0,
        float,
        "Distance the terminal assist inserts by. Non-zero is an assisted run.",
        minimum=0.0, maximum=0.05,
    ),
    Parameter(
        "robotwin_denoiser", "PANTHERA_ROBOTWIN_DENOISER", None, str,
        "Optional observation denoiser selection.",
    ),
    Parameter(
        "eval_trace_dir", "PANTHERA_EVAL_TRACE_DIR", None, str,
        "Where per-step evaluation traces are written; diagnostic only.",
    ),
)

_BY_NAME = {parameter.name: parameter for parameter in PARAMETERS}


@dataclass(frozen=True)
class SimConfig:
    """Fully materialised simulation configuration.

    Every value is present, whether it came from the environment or a default,
    so a snapshot of this object is enough to say what physics a dataset was
    collected under.
    """

    values: Mapping[str, Any]
    sources: Mapping[str, str]

    def __getitem__(self, name: str) -> Any:
        if name not in self.values:
            raise KeyError(f"unknown simulation parameter: {name}")
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    @property
    def geometry(self) -> dict:
        return {
            "cylinder_radius_m": CYLINDER_RADIUS_M,
            "cylinder_half_height_m": CYLINDER_HALF_HEIGHT_M,
            "table_height_m": TABLE_HEIGHT_M,
            "spawn_clearance_m": SPAWN_CLEARANCE_M,
        }

    def snapshot(self) -> dict:
        """The form a dataset or a run record stores."""
        return {
            "schema_version": 1,
            "geometry": self.geometry,
            "values": dict(self.values),
            "sources": dict(self.sources),
        }

    def difference(self, other: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
        """Report every parameter whose value differs from a stored snapshot."""
        stored = other.get("values", other)
        names = set(self.values) | set(stored)
        return {
            name: (stored.get(name), self.values.get(name))
            for name in sorted(names)
            if stored.get(name) != self.values.get(name)
        }


def resolve(environ: Optional[Mapping[str, str]] = None) -> SimConfig:
    """Materialise every declared parameter from an environment."""
    source_env = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for parameter in PARAMETERS:
        raw = source_env.get(parameter.env) if parameter.env else None
        value = parameter.parse(raw)
        if value is not None:
            parameter.validate(value)
        values[parameter.name] = value
        sources[parameter.name] = "environment" if raw is not None else "default"
    return SimConfig(values=values, sources=sources)


def from_snapshot(snapshot: Mapping[str, Any]) -> SimConfig:
    """Rebuild a configuration a dataset recorded, validating it on the way in."""
    stored = snapshot.get("values")
    if not isinstance(stored, Mapping):
        raise SimConfigError("snapshot has no 'values' mapping")
    unknown = sorted(set(stored) - set(_BY_NAME))
    if unknown:
        raise SimConfigError(f"snapshot declares unknown parameters: {unknown}")
    missing = sorted(set(_BY_NAME) - set(stored))
    if missing:
        raise SimConfigError(f"snapshot is missing parameters: {missing}")
    for name, value in stored.items():
        if value is not None:
            _BY_NAME[name].validate(value)
    return SimConfig(
        values=dict(stored),
        sources=dict(snapshot.get("sources", {name: "snapshot" for name in stored})),
    )


def main() -> int:
    """Print the resolved configuration, for use in a run record."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true",
                        help="list every declared parameter instead of resolving")
    cli = parser.parse_args()
    if cli.declare:
        for parameter in PARAMETERS:
            bound = []
            if parameter.minimum is not None:
                bound.append(f"{'>' if parameter.exclusive_minimum else '>='}{parameter.minimum}")
            if parameter.maximum is not None:
                bound.append(f"<={parameter.maximum}")
            if parameter.choices is not None:
                bound.append(f"in {parameter.choices}")
            print(f"{parameter.name:38s} {str(parameter.env or '-'):46s} "
                  f"默认={parameter.default!r:>10} {' '.join(bound)}")
        return 0
    print(json.dumps(resolve().snapshot(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
