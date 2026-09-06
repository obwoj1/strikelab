"""Draws the analysis over a frame.

The output is meant to be reviewable at a glance by someone who was not
watching the terminal: goal frame and placement grid, the ball's trail, the
striker's skeleton, live ball speed, and a verdict card that holds on screen
for a beat after each shot resolves.
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:  # pragma: no cover - exercised whenever video is used
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

from ..config import RenderConfig
from ..engine import FrameResult, Shot
from ..geometry import GoalCalibration
from ..types import COCO_SKELETON

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX

VERDICT_COLORS: dict[str, tuple[int, int, int]] = {
    "GOAL": (80, 220, 100),
    "SAVED": (240, 180, 60),
    "WOODWORK": (60, 200, 240),
    "BLOCKED": (150, 150, 150),
    "OFF_TARGET": (80, 90, 230),
    "UNRESOLVED": (170, 170, 170),
}

GRID = (170, 170, 170)
GOAL_LINE = (120, 230, 255)
TRAIL = (255, 210, 120)
SKELETON = (250, 210, 120)
PANEL = (28, 24, 20)


class Annotator:
    def __init__(
        self,
        calibration: GoalCalibration,
        config: RenderConfig | None = None,
        *,
        fps: float = 30.0,
    ) -> None:
        if cv2 is None:  # pragma: no cover
            raise RuntimeError(
                "OpenCV is required for annotation. pip install strikelab[video]"
            )
        self.calibration = calibration
        self.config = config or RenderConfig()
        self.fps = fps
        self._trail: deque[tuple[int, int]] = deque(maxlen=self.config.trail_frames)
        self._banner: tuple[Shot, int] | None = None
        self._shots: list[Shot] = []

    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        canvas = frame.copy()

        if self.config.draw_goal_mesh:
            self._draw_goal_frame(canvas)
        if self.config.draw_zone_grid:
            self._draw_zone_grid(canvas)
        if self.config.draw_skeleton:
            self._draw_skeletons(canvas, result)

        self._draw_people(canvas, result)
        self._draw_ball(canvas, result)

        if result.completed_shot is not None:
            self._shots.append(result.completed_shot)
            self._banner = (result.completed_shot, result.frame_index)

        if self.config.draw_hud:
            self._draw_hud(canvas, result)
        self._draw_banner(canvas, result)

        return canvas

    # ------------------------------------------------------------------

    def _draw_goal_frame(self, canvas: np.ndarray) -> None:
        corners = np.array(self.calibration.mouth_polygon(), dtype=np.int32)
        cv2.polylines(canvas, [corners], True, GOAL_LINE, 2, cv2.LINE_AA)

    def _draw_zone_grid(self, canvas: np.ndarray) -> None:
        for start, end in self.calibration.zone_grid_segments():
            cv2.line(canvas, _pt(start), _pt(end), GRID, 1, cv2.LINE_AA)

    def _draw_skeletons(self, canvas: np.ndarray, result: FrameResult) -> None:
        for pose in result.detections.poses:
            for a, b in COCO_SKELETON:
                pa, pb = pose.point(a), pose.point(b)
                if pa and pb:
                    cv2.line(canvas, _pt(pa), _pt(pb), SKELETON, 2, cv2.LINE_AA)
            for name in ("left_ankle", "right_ankle"):
                point = pose.point(name)
                if point:
                    cv2.circle(canvas, _pt(point), 4, SKELETON, -1, cv2.LINE_AA)

    def _draw_people(self, canvas: np.ndarray, result: FrameResult) -> None:
        for keeper in result.detections.goalkeepers:
            cv2.rectangle(
                canvas,
                _pt((keeper.x1, keeper.y1)),
                _pt((keeper.x2, keeper.y2)),
                (60, 200, 240),
                2,
            )
            cv2.putText(
                canvas, "GK", _pt((keeper.x1, keeper.y1 - 6)), FONT, 0.45, (60, 200, 240), 1, cv2.LINE_AA
            )

    def _draw_ball(self, canvas: np.ndarray, result: FrameResult) -> None:
        if result.ball is None:
            self._trail.clear()
            return

        point = _pt(result.ball.position)
        if self.config.draw_trail:
            self._trail.append(point)
            for index in range(1, len(self._trail)):
                weight = index / len(self._trail)
                cv2.line(
                    canvas,
                    self._trail[index - 1],
                    self._trail[index],
                    TRAIL,
                    max(1, int(1 + 3 * weight)),
                    cv2.LINE_AA,
                )

        colour = (90, 250, 250) if result.inside_mouth else (255, 255, 255)
        cv2.circle(canvas, point, 9, colour, 2, cv2.LINE_AA)
        cv2.circle(canvas, point, 2, colour, -1, cv2.LINE_AA)

        if result.contact is not None:
            cv2.circle(canvas, _pt(result.contact[1]), 13, (80, 220, 100), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------

    def _draw_hud(self, canvas: np.ndarray, result: FrameResult) -> None:
        height, width = canvas.shape[:2]
        panel_h = 62
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_h), PANEL, -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

        goals = sum(1 for s in self._shots if s.verdict == "GOAL")
        on_target = sum(1 for s in self._shots if s.on_target)
        speed = result.ball.speed_mps * 3.6 if result.ball else 0.0

        cv2.putText(canvas, "StrikeLab", (14, 26), FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"state {result.state}",
            (14, 48),
            FONT,
            0.45,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

        stats = [
            f"shots {len(self._shots)}",
            f"on target {on_target}",
            f"goals {goals}",
            f"ball {speed:5.1f} km/h",
        ]
        x = 190
        for item in stats:
            cv2.putText(canvas, item, (x, 38), FONT, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
            x += 170

    def _draw_banner(self, canvas: np.ndarray, result: FrameResult) -> None:
        if self._banner is None:
            return
        shot, shown_at = self._banner
        hold = int(self.fps * 2.0)
        if result.frame_index - shown_at > hold:
            self._banner = None
            return

        height, width = canvas.shape[:2]
        colour = VERDICT_COLORS.get(shot.verdict, (200, 200, 200))
        box_w, box_h = 430, 108
        x0, y0 = 16, height - box_h - 16

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), PANEL, -1)
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), colour, 2)

        cv2.putText(
            canvas, shot.verdict, (x0 + 14, y0 + 34), FONT, 0.86, colour, 2, cv2.LINE_AA
        )

        speed = shot.peak_speed_mps * 3.6
        placement = shot.zone or "off target"
        quality = shot.quality["score"] if shot.quality else 0.0
        lines = [
            f"{speed:.0f} km/h   {placement}",
            f"quality {quality}   {shot.confidence}",
        ]
        if shot.distance_m is not None:
            lines.append(f"{shot.distance_m:.1f} m out, {shot.angle_deg:.0f} deg of goal")
        elif shot.notes:
            lines.append(shot.notes[0][:52])

        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (x0 + 14, y0 + 58 + index * 21),
                FONT,
                0.48,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )


def _pt(point: tuple[float, float]) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))
