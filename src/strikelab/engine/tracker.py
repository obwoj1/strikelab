"""Ball track: association, smoothing, and velocity in metres per second.

A soccer ball is small, fast and frequently occluded by legs. Two things matter
more than any clever filter: rejecting detections that are too far from where
the ball should be, and coasting through short gaps rather than dropping the
track and re-arming the state machine mid-shot.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..config import TrackerConfig
from ..constants import BALL_DIAMETER_M
from ..types import Box, Point


@dataclass
class BallSample:
    frame_index: int
    position: Point
    raw_position: Point
    velocity: Point  # pixels per frame
    speed_mps: float
    box: Box | None
    interpolated: bool


class BallTracker:
    """Single-target tracker with a constant-velocity prediction.

    Speed is reported in metres per second by dividing pixel motion by the
    image scale at the goal plane. That is exact for a ball travelling in the
    goal plane and optimistic for one nearer the camera, which is why the
    engine also records the scale it used.
    """

    def __init__(
        self,
        config: TrackerConfig,
        *,
        pixels_per_metre: float,
        fps: float,
        goal_width_px: float,
    ) -> None:
        if pixels_per_metre <= 0:
            raise ValueError("pixels_per_metre must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.config = config
        self.pixels_per_metre = pixels_per_metre
        self.fps = fps
        self.max_association_px = (
            goal_width_px * config.max_association_dist_goal_widths
        )
        self._history: deque[BallSample] = deque(maxlen=256)
        self._missing = 0
        self._smoothed: Point | None = None

    @property
    def history(self) -> list[BallSample]:
        return list(self._history)

    @property
    def last(self) -> BallSample | None:
        return self._history[-1] if self._history else None

    @property
    def active(self) -> bool:
        return bool(self._history) and self._missing <= self.config.max_gap_frames

    def reset(self) -> None:
        self._history.clear()
        self._missing = 0
        self._smoothed = None

    def _predict(self) -> Point | None:
        last = self.last
        if last is None:
            return None
        return (
            last.position[0] + last.velocity[0],
            last.position[1] + last.velocity[1],
        )

    def _mean_velocity(self) -> Point:
        window = list(self._history)[-self.config.velocity_window :]
        if len(window) < 2:
            return (0.0, 0.0)
        first, last = window[0], window[-1]
        span = max(1, last.frame_index - first.frame_index)
        return (
            (last.position[0] - first.position[0]) / span,
            (last.position[1] - first.position[1]) / span,
        )

    def update(self, frame_index: int, ball: Box | None) -> BallSample | None:
        """Feed one frame. Returns the current sample, or None if there is no track."""
        candidate: Point | None = None

        if ball is not None and ball.confidence >= self.config.min_confidence:
            candidate = ball.centre
            prediction = self._predict()
            if prediction is not None:
                distance = math.hypot(
                    candidate[0] - prediction[0], candidate[1] - prediction[1]
                )
                # A jump far beyond the prediction is almost always a second
                # ball on the pitch, not our ball teleporting.
                if distance > self.max_association_px:
                    candidate = None

        if candidate is None:
            self._missing += 1
            if not self._history or self._missing > self.config.max_gap_frames:
                if self._missing > self.config.max_gap_frames:
                    self.reset()
                return None
            # Coast on the last known velocity so a shot survives a leg
            # crossing in front of the ball.
            coasted = self._predict()
            if coasted is None:
                return None
            sample = BallSample(
                frame_index=frame_index,
                position=coasted,
                raw_position=coasted,
                velocity=self._history[-1].velocity,
                speed_mps=self._history[-1].speed_mps,
                box=None,
                interpolated=True,
            )
            self._history.append(sample)
            self._smoothed = coasted
            return sample

        self._missing = 0
        alpha = self.config.position_alpha
        if self._smoothed is None:
            smoothed = candidate
        else:
            smoothed = (
                alpha * candidate[0] + (1.0 - alpha) * self._smoothed[0],
                alpha * candidate[1] + (1.0 - alpha) * self._smoothed[1],
            )
        self._smoothed = smoothed

        sample = BallSample(
            frame_index=frame_index,
            position=smoothed,
            raw_position=candidate,
            velocity=(0.0, 0.0),
            speed_mps=0.0,
            box=ball,
            interpolated=False,
        )
        self._history.append(sample)

        velocity = self._mean_velocity()
        speed_px_per_frame = math.hypot(*velocity)
        speed_mps = speed_px_per_frame * self.fps / self.pixels_per_metre
        self._history[-1] = BallSample(
            frame_index=sample.frame_index,
            position=sample.position,
            raw_position=sample.raw_position,
            velocity=velocity,
            speed_mps=speed_mps,
            box=ball,
            interpolated=False,
        )
        return self._history[-1]

    def ball_diameter_px(self) -> float:
        """Expected ball size at the goal plane, used for contact thresholds."""
        return BALL_DIAMETER_M * self.pixels_per_metre

    def recent_speed(self, frames: int) -> float:
        """Median speed over the last N samples, ignoring coasted frames."""
        window = [s.speed_mps for s in list(self._history)[-frames:] if not s.interpolated]
        if not window:
            return 0.0
        window.sort()
        middle = len(window) // 2
        if len(window) % 2:
            return window[middle]
        return (window[middle - 1] + window[middle]) / 2.0
