"""Goal-plane geometry.

The basketball evaluator calibrates on a single scalar: the rim is 18 inches
wide, so one pixel is N metres. That works because a rim is small and roughly
fronto-parallel.

A goal is 7.32 m wide and you almost always view it at an angle, so a single
scale factor is not enough. Instead we fit a homography from the image to the
plane of the goal mouth. That buys three things a scalar cannot:

  * a true "is this inside the frame of the goal?" test, at any camera angle,
  * placement in metres from the left post and the ground, hence the 3x3
    coaching grid,
  * a perspective-correct distance from the posts and the crossbar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .constants import (
    GOAL_HEIGHT_M,
    GOAL_WIDTH_M,
    ZONE_COLUMNS,
    ZONE_NAMES,
    ZONE_ROWS,
)

Point = tuple[float, float]


class CalibrationError(ValueError):
    """Raised when the supplied goal corners cannot form a usable homography."""


def _as_array(points: Sequence[Point]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (4, 2):
        raise CalibrationError(
            f"expected exactly 4 (x, y) corners, got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise CalibrationError("corner coordinates must all be finite")
    return array


def order_goal_corners(points: Sequence[Point]) -> np.ndarray:
    """Sort four unordered corners into (top-left, top-right, bottom-right, bottom-left).

    Image y grows downward, so "top" is the crossbar. Ordering by y splits the
    crossbar pair from the post-base pair; ordering each pair by x then fixes
    left from right.
    """
    array = _as_array(points)
    by_y = array[np.argsort(array[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct linear transform for a 4-point homography.

    Implemented here rather than via cv2.getPerspectiveTransform so the
    geometry module stays import-light and unit-testable without OpenCV.
    """
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    matrix = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(matrix)
    h = vh[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise CalibrationError("degenerate corner configuration (collinear points?)")
    return h / h[2, 2]


def _apply(h: np.ndarray, point: Point) -> Point:
    vector = h @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if abs(vector[2]) < 1e-12:
        raise CalibrationError("point projects to infinity under this homography")
    return (float(vector[0] / vector[2]), float(vector[1] / vector[2]))


@dataclass(frozen=True)
class GoalCalibration:
    """Maps image pixels to the plane of the goal mouth.

    Goal-plane coordinates are in metres: X runs 0 at the left post to 7.32 at
    the right post, Y runs 0 at the ground to 2.44 at the crossbar. Both are
    from the camera's point of view, which is the frame a coach describes a
    finish in ("bottom left corner").
    """

    corners_tl_tr_br_bl: tuple[Point, Point, Point, Point]
    width_m: float = GOAL_WIDTH_M
    height_m: float = GOAL_HEIGHT_M
    _to_goal: np.ndarray = None  # type: ignore[assignment]
    _to_image: np.ndarray = None  # type: ignore[assignment]

    @classmethod
    def from_corners(
        cls,
        points: Sequence[Point],
        *,
        width_m: float = GOAL_WIDTH_M,
        height_m: float = GOAL_HEIGHT_M,
        assume_ordered: bool = False,
    ) -> "GoalCalibration":
        if width_m <= 0 or height_m <= 0:
            raise CalibrationError("goal dimensions must be positive")
        ordered = _as_array(points) if assume_ordered else order_goal_corners(points)

        # Destination corners in goal-plane metres, matching the source order:
        # top-left, top-right, bottom-right, bottom-left.
        dst = np.array(
            [
                [0.0, height_m],
                [width_m, height_m],
                [width_m, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )

        to_goal = _homography(ordered, dst)
        to_image = np.linalg.inv(to_goal)
        corners = tuple((float(x), float(y)) for x, y in ordered)
        instance = cls(
            corners_tl_tr_br_bl=corners,  # type: ignore[arg-type]
            width_m=width_m,
            height_m=height_m,
        )
        object.__setattr__(instance, "_to_goal", to_goal)
        object.__setattr__(instance, "_to_image", to_image)
        instance._self_check()
        return instance

    def _self_check(self) -> None:
        """Round-trip the corners; a bad quad shows up here rather than mid-video."""
        for corner in self.corners_tl_tr_br_bl:
            gx, gy = self.to_goal_plane(corner)
            if not (-0.05 <= gx <= self.width_m + 0.05):
                raise CalibrationError("corner did not map onto the goal plane")
            if not (-0.05 <= gy <= self.height_m + 0.05):
                raise CalibrationError("corner did not map onto the goal plane")

    def to_goal_plane(self, point: Point) -> Point:
        """Image pixel -> (metres from left post, metres above ground)."""
        return _apply(self._to_goal, point)

    def to_image(self, goal_point: Point) -> Point:
        """Goal-plane metres -> image pixel."""
        return _apply(self._to_image, goal_point)

    def contains(self, point: Point, *, margin_m: float = 0.0) -> bool:
        """Does this image point project inside the frame of the goal?"""
        x, y = self.to_goal_plane(point)
        return (
            -margin_m <= x <= self.width_m + margin_m
            and -margin_m <= y <= self.height_m + margin_m
        )

    def inside_goal_plane(self, goal_point: Point, *, margin_m: float = 0.0) -> bool:
        x, y = goal_point
        return (
            -margin_m <= x <= self.width_m + margin_m
            and -margin_m <= y <= self.height_m + margin_m
        )

    def distance_to_frame_m(self, goal_point: Point) -> float:
        """Signed distance to the nearest post or the crossbar, in metres.

        Positive inside the mouth, negative outside. This is what separates a
        goal from woodwork from a shot that missed by a yard.
        """
        x, y = goal_point
        inside = self.inside_goal_plane(goal_point)
        if inside:
            return float(min(x, self.width_m - x, y, self.height_m - y))
        dx = max(0.0, -x, x - self.width_m)
        dy = max(0.0, -y, y - self.height_m)
        return -float(math.hypot(dx, dy))

    def zone_of(self, goal_point: Point) -> str | None:
        """Which cell of the 3x3 coaching grid a finish landed in."""
        if not self.inside_goal_plane(goal_point):
            return None
        x, y = goal_point
        column = min(
            ZONE_COLUMNS - 1, max(0, int(x / self.width_m * ZONE_COLUMNS))
        )
        row = min(ZONE_ROWS - 1, max(0, int(y / self.height_m * ZONE_ROWS)))
        return ZONE_NAMES[row * ZONE_COLUMNS + column]

    def miss_description(self, goal_point: Point) -> str:
        """Plain words for where an off-target shot went."""
        x, y = goal_point
        parts: list[str] = []
        if y > self.height_m:
            parts.append(f"{y - self.height_m:.1f} m over")
        if x < 0:
            parts.append(f"{-x:.1f} m wide left")
        elif x > self.width_m:
            parts.append(f"{x - self.width_m:.1f} m wide right")
        if not parts:
            parts.append("into the ground")
        return " and ".join(parts)

    def pixels_per_metre(self) -> float:
        """Approximate image scale at the goal plane, from the crossbar width."""
        (tlx, tly), (trx, try_), _, _ = self.corners_tl_tr_br_bl
        return float(math.hypot(trx - tlx, try_ - tly) / self.width_m)

    def mouth_polygon(self) -> list[Point]:
        return [tuple(map(float, corner)) for corner in self.corners_tl_tr_br_bl]

    def zone_grid_segments(self) -> list[tuple[Point, Point]]:
        """Image-space line segments for drawing the 3x3 grid on the goal."""
        segments: list[tuple[Point, Point]] = []
        for index in range(1, ZONE_COLUMNS):
            x = self.width_m * index / ZONE_COLUMNS
            segments.append((self.to_image((x, 0.0)), self.to_image((x, self.height_m))))
        for index in range(1, ZONE_ROWS):
            y = self.height_m * index / ZONE_ROWS
            segments.append((self.to_image((0.0, y)), self.to_image((self.width_m, y))))
        return segments

    def to_json(self) -> dict[str, object]:
        return {
            "corners_tl_tr_br_bl": [list(corner) for corner in self.corners_tl_tr_br_bl],
            "width_m": self.width_m,
            "height_m": self.height_m,
            "pixels_per_metre": self.pixels_per_metre(),
        }


def centroid(points: Iterable[Point]) -> Point:
    array = np.asarray(list(points), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot take the centroid of an empty set")
    return (float(array[:, 0].mean()), float(array[:, 1].mean()))


def goal_direction(from_point: Point, calibration: GoalCalibration) -> Point:
    """Unit vector in image space pointing from a point at the goal centre."""
    target = calibration.to_image((calibration.width_m / 2.0, calibration.height_m / 2.0))
    dx = target[0] - from_point[0]
    dy = target[1] - from_point[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return (0.0, 0.0)
    return (dx / norm, dy / norm)


def cosine(a: Point, b: Point) -> float:
    """Cosine of the angle between two image-space vectors; 0 if either is null."""
    na = math.hypot(*a)
    nb = math.hypot(*b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a[0] * b[0] + a[1] * b[1]) / (na * nb))
