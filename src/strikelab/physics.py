"""Speed, distance, shot angle and a transparent shot-quality score.

Basketball's evaluator leans on a single physical trick: after the ball is
occluded by the net it reappears at 10-90% of freefall speed, and that ratio is
the make/miss signal. Soccer does not need that trick, because a goal is an
open plane rather than a hole you fall through. What soccer needs instead is
distance and angle, because a shot's difficulty depends on where it was struck
from far more than a basketball shot does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import (
    GOAL_WIDTH_M,
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_WIDTH_M,
    SIX_YARD_BOX_DEPTH_M,
    SIX_YARD_BOX_WIDTH_M,
    ZONE_DIFFICULTY,
)
from .geometry import CalibrationError, Point, _homography, _apply

MPS_TO_KMH = 3.6
MPS_TO_MPH = 2.2369362920544


def mps_to_kmh(speed_mps: float) -> float:
    return speed_mps * MPS_TO_KMH


def mps_to_mph(speed_mps: float) -> float:
    return speed_mps * MPS_TO_MPH


# Ground-plane presets: image points are supplied in this order, and these are
# the pitch coordinates they correspond to. Origin is the centre of the goal
# line; +x runs toward the right post, +y runs out onto the pitch.
GROUND_PRESETS: dict[str, tuple[tuple[float, float], ...]] = {
    # Corners of the six-yard box, starting at the left goal-line corner and
    # going clockwise as seen from behind the goal.
    "six-yard-box": (
        (-SIX_YARD_BOX_WIDTH_M / 2, 0.0),
        (SIX_YARD_BOX_WIDTH_M / 2, 0.0),
        (SIX_YARD_BOX_WIDTH_M / 2, SIX_YARD_BOX_DEPTH_M),
        (-SIX_YARD_BOX_WIDTH_M / 2, SIX_YARD_BOX_DEPTH_M),
    ),
    "penalty-area": (
        (-PENALTY_AREA_WIDTH_M / 2, 0.0),
        (PENALTY_AREA_WIDTH_M / 2, 0.0),
        (PENALTY_AREA_WIDTH_M / 2, PENALTY_AREA_DEPTH_M),
        (-PENALTY_AREA_WIDTH_M / 2, PENALTY_AREA_DEPTH_M),
    ),
}


@dataclass(frozen=True)
class GroundCalibration:
    """Optional image-to-pitch homography for the ground plane.

    Without this we can still say where a shot finished on the goal mouth and
    how fast it was travelling in goal-widths per second, but we cannot say it
    was struck from 18 metres. Distance changes shot quality more than anything
    else, so supplying four ground points is the single highest-value piece of
    setup a user can do.
    """

    image_points: tuple[Point, Point, Point, Point]
    pitch_points: tuple[Point, Point, Point, Point]
    _to_pitch: np.ndarray = None  # type: ignore[assignment]

    @classmethod
    def from_points(
        cls,
        image_points: Sequence[Point],
        pitch_points: Sequence[Point],
    ) -> "GroundCalibration":
        src = np.asarray(image_points, dtype=np.float64)
        dst = np.asarray(pitch_points, dtype=np.float64)
        if src.shape != (4, 2) or dst.shape != (4, 2):
            raise CalibrationError("ground calibration needs exactly 4 point pairs")
        matrix = _homography(src, dst)
        instance = cls(
            image_points=tuple(map(tuple, src)),  # type: ignore[arg-type]
            pitch_points=tuple(map(tuple, dst)),  # type: ignore[arg-type]
        )
        object.__setattr__(instance, "_to_pitch", matrix)
        return instance

    @classmethod
    def from_preset(cls, name: str, image_points: Sequence[Point]) -> "GroundCalibration":
        try:
            pitch = GROUND_PRESETS[name]
        except KeyError as error:
            options = ", ".join(sorted(GROUND_PRESETS))
            raise CalibrationError(f"unknown ground preset {name!r}; try one of: {options}") from error
        return cls.from_points(image_points, pitch)

    def to_pitch(self, point: Point) -> Point:
        """Image pixel on the ground -> (metres across, metres from goal line)."""
        return _apply(self._to_pitch, point)

    def distance_to_goal_m(self, point: Point) -> float:
        x, y = self.to_pitch(point)
        return float(math.hypot(x, y))

    def to_json(self) -> dict[str, object]:
        return {
            "image_points": [list(p) for p in self.image_points],
            "pitch_points": [list(p) for p in self.pitch_points],
        }


def shot_angle_deg(pitch_point: Point, goal_width_m: float = GOAL_WIDTH_M) -> float:
    """The angle the goal mouth subtends from where the shot was struck.

    This is the classic "shooting angle" used in every public expected-goals
    model. A shot from the penalty spot sees a wide goal; the same distance out
    by the touchline sees almost none of it.
    """
    x, y = pitch_point
    half = goal_width_m / 2.0
    if y <= 0.0:
        # On or behind the goal line: no meaningful angle.
        return 0.0
    left = math.atan2(-half - x, y)
    right = math.atan2(half - x, y)
    return float(abs(math.degrees(right - left)))


def freefall_drop_m(seconds: float, gravity: float = 9.81) -> float:
    return 0.5 * gravity * seconds * seconds


@dataclass(frozen=True)
class ShotQuality:
    """A transparent, hand-tuned shot score. Explicitly not a trained xG model.

    Every term is inspectable and every input is reported alongside it, so a
    coach can disagree with the number and still use the components. Calling it
    xG would imply it was fitted to a shot database; it was not.
    """

    score: float
    distance_term: float
    angle_term: float
    placement_term: float
    speed_term: float
    inputs: dict[str, float | str | None]

    def to_json(self) -> dict[str, object]:
        return {
            "score": round(self.score, 4),
            "distance_term": round(self.distance_term, 4),
            "angle_term": round(self.angle_term, 4),
            "placement_term": round(self.placement_term, 4),
            "speed_term": round(self.speed_term, 4),
            "inputs": self.inputs,
        }


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def shot_quality(
    *,
    distance_m: float | None,
    angle_deg: float | None,
    zone: str | None,
    speed_mps: float | None,
    on_target: bool,
) -> ShotQuality:
    """Score a shot from 0 to 1 on how likely that attempt was to be scored.

    Shape of the model, in order of influence:

      * distance   - a logistic falling off from roughly 6 m out, the dominant
                     term in every published expected-goals model,
      * angle      - how much of the goal was actually visible from there,
      * placement  - corner finishes beat central ones (only known on target),
      * speed      - a hard strike gives the keeper less time, with heavily
                     diminishing returns past about 25 m/s.

    Missing inputs fall back to a neutral prior rather than zero, and the
    fallback is recorded in `inputs` so nothing is silently invented.
    """
    used_distance = distance_m if distance_m is not None else 16.0
    used_angle = angle_deg if angle_deg is not None else 25.0

    # Distance: ~0.9 at 5 m, ~0.35 at 12 m, ~0.08 at 25 m.
    distance_term = _logistic(1.6 - 0.22 * used_distance) * 1.35
    distance_term = min(1.0, distance_term)

    # Angle: 0 at no visible goal, saturating around 60 degrees.
    angle_term = min(1.0, used_angle / 55.0)

    placement_term = ZONE_DIFFICULTY.get(zone or "", 1.0) if on_target else 0.55

    if speed_mps is None:
        speed_term = 1.0
    else:
        speed_term = 0.75 + 0.35 * min(1.0, max(0.0, (speed_mps - 8.0) / 17.0))

    raw = distance_term * (0.45 + 0.55 * angle_term) * placement_term * speed_term
    score = float(min(0.99, max(0.005, raw)))

    return ShotQuality(
        score=score,
        distance_term=distance_term,
        angle_term=angle_term,
        placement_term=placement_term,
        speed_term=speed_term,
        inputs={
            "distance_m": distance_m,
            "angle_deg": angle_deg,
            "zone": zone,
            "speed_mps": speed_mps,
            "distance_source": "measured" if distance_m is not None else "prior(16m)",
            "angle_source": "measured" if angle_deg is not None else "prior(25deg)",
        },
    )
