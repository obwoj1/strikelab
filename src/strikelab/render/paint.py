"""Paints the synthetic scene into actual pixels.

Kept apart from `sources.synthetic`, which stays pure geometry so the tests can
drive the engine without importing OpenCV.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised whenever video is used
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

from ..geometry import GoalCalibration
from ..sources.synthetic import SceneFrame
from ..types import COCO_SKELETON

PITCH = (58, 104, 48)
PITCH_STRIPE = (64, 116, 54)
GOAL_WHITE = (238, 238, 238)
NET = (150, 158, 150)
BALL = (250, 250, 250)
BALL_EDGE = (30, 30, 30)
PLAYER = (60, 90, 220)
KEEPER = (60, 200, 240)


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            "OpenCV is required for video output. Install with: pip install strikelab[video]"
        )


def paint_frame(
    scene_frame: SceneFrame,
    calibration: GoalCalibration,
    width: int,
    height: int,
) -> np.ndarray:
    """Render one synthetic frame as a BGR image."""
    _require_cv2()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = PITCH

    # Mown stripes, purely so the demo video does not look like a test card.
    for index in range(0, height, 56):
        if (index // 56) % 2 == 0:
            canvas[index : index + 56, :] = PITCH_STRIPE

    _draw_goal(canvas, calibration)

    # Keeper.
    if scene_frame.keeper_centre is not None:
        kx, ky = (int(v) for v in scene_frame.keeper_centre)
        cv2.rectangle(canvas, (kx - 20, ky - 68), (kx + 20, ky + 6), KEEPER, -1)
        cv2.circle(canvas, (kx, ky - 82), 13, KEEPER, -1)

    # Striker.
    keypoints = scene_frame.player_keypoints
    for a, b in COCO_SKELETON:
        if a in keypoints and b in keypoints:
            cv2.line(
                canvas,
                tuple(int(v) for v in keypoints[a]),  # type: ignore[arg-type]
                tuple(int(v) for v in keypoints[b]),  # type: ignore[arg-type]
                PLAYER,
                5,
                cv2.LINE_AA,
            )
    if "nose" in keypoints:
        cv2.circle(canvas, tuple(int(v) for v in keypoints["nose"]), 13, PLAYER, -1)  # type: ignore[arg-type]

    # Ball.
    ball = scene_frame.detections.ball
    if ball is not None:
        cx, cy = ball.centre
        radius = max(3, int(scene_frame.ball_radius_px))
        cv2.circle(canvas, (int(cx), int(cy)), radius, BALL, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(cx), int(cy)), radius, BALL_EDGE, 1, cv2.LINE_AA)

    return canvas


def _draw_goal(canvas: np.ndarray, calibration: GoalCalibration) -> None:
    corners = np.array(calibration.mouth_polygon(), dtype=np.int32)

    # Net: a light mesh inside the mouth.
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [corners], (86, 96, 86))
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    for index in range(1, 12):
        t = index / 12.0
        top = calibration.to_image((calibration.width_m * t, calibration.height_m))
        bottom = calibration.to_image((calibration.width_m * t, 0.0))
        cv2.line(canvas, _pt(top), _pt(bottom), NET, 1, cv2.LINE_AA)
    for index in range(1, 6):
        t = index / 6.0
        left = calibration.to_image((0.0, calibration.height_m * t))
        right = calibration.to_image((calibration.width_m, calibration.height_m * t))
        cv2.line(canvas, _pt(left), _pt(right), NET, 1, cv2.LINE_AA)

    # Frame: posts and crossbar.
    cv2.polylines(canvas, [corners], True, GOAL_WHITE, 7, cv2.LINE_AA)


def _pt(point: tuple[float, float]) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))
