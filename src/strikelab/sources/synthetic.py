"""A scripted synthetic training-ground scene.

This exists for two reasons:

  * `strikelab demo` runs the entire pipeline end to end with no model weights,
    no API key and no network, so anyone can see the output shape immediately,
  * the test suite drives the real engine with exactly known ground truth, so a
    regression in the state machine fails a test rather than quietly changing a
    verdict on somebody's footage.

It is not a simulator. Ball motion is scripted in image space with a light arc;
it is calibrated to produce plausible pixel speeds, not to model aerodynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np

from ..geometry import GoalCalibration
from ..types import Box, FrameDetections, Point, Pose

Outcome = Literal["goal", "saved", "wide", "over", "woodwork"]

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# A goal viewed from in front and slightly to the side, as a training-ground
# camera on a tripod behind the shooter would see it.
GOAL_CORNERS: tuple[Point, Point, Point, Point] = (
    (430.0, 196.0),  # top-left  (crossbar, left post)
    (868.0, 188.0),  # top-right
    (902.0, 372.0),  # bottom-right (right post base)
    (398.0, 380.0),  # bottom-left
)


@dataclass
class ShotScript:
    """One scripted attempt."""

    outcome: Outcome
    # Where the ball is struck from, in image pixels.
    start: Point
    # Aim point in goal-plane metres (x from left post, y above ground).
    target_goal_m: Point
    approach_frames: int = 26
    flight_frames: int = 12
    settle_frames: int = 14
    label: str = ""


@dataclass
class SceneFrame:
    index: int
    detections: FrameDetections
    # Extra bookkeeping the renderer uses to draw the synthetic image.
    player_keypoints: dict[str, Point] = field(default_factory=dict)
    keeper_centre: Point | None = None
    ball_radius_px: float = 9.0
    caption: str = ""


DEFAULT_SCRIPTS: tuple[ShotScript, ...] = (
    ShotScript(
        outcome="goal",
        start=(596.0, 628.0),
        target_goal_m=(0.85, 0.55),
        label="low into the bottom-left corner",
    ),
    ShotScript(
        outcome="saved",
        start=(700.0, 648.0),
        target_goal_m=(3.66, 1.15),
        label="straight at the keeper",
    ),
    ShotScript(
        outcome="wide",
        start=(520.0, 640.0),
        target_goal_m=(-1.45, 1.05),
        label="dragged wide of the left post",
    ),
    ShotScript(
        outcome="goal",
        start=(742.0, 612.0),
        target_goal_m=(6.55, 1.92),
        label="into the top-right postage stamp",
    ),
    ShotScript(
        outcome="over",
        start=(636.0, 656.0),
        target_goal_m=(3.66, 3.35),
        label="ballooned over the crossbar",
    ),
)


def default_calibration() -> GoalCalibration:
    return GoalCalibration.from_corners(GOAL_CORNERS, assume_ordered=True)


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _stick_figure(hip: Point, scale: float, stride: float) -> dict[str, Point]:
    """A crude but anatomically ordered COCO-17 skeleton.

    Only the joints the engine actually reads need to be right; the rest exist
    so the rendered overlay looks like a person rather than a scatter plot.
    """
    hx, hy = hip
    swing = math.sin(stride) * 0.42 * scale
    lift = abs(math.cos(stride)) * 0.18 * scale
    return {
        "nose": (hx, hy - 1.55 * scale),
        "left_eye": (hx - 0.06 * scale, hy - 1.60 * scale),
        "right_eye": (hx + 0.06 * scale, hy - 1.60 * scale),
        "left_ear": (hx - 0.12 * scale, hy - 1.58 * scale),
        "right_ear": (hx + 0.12 * scale, hy - 1.58 * scale),
        "left_shoulder": (hx - 0.26 * scale, hy - 1.24 * scale),
        "right_shoulder": (hx + 0.26 * scale, hy - 1.24 * scale),
        "left_elbow": (hx - 0.42 * scale, hy - 0.92 * scale),
        "right_elbow": (hx + 0.42 * scale, hy - 0.92 * scale),
        "left_wrist": (hx - 0.50 * scale, hy - 0.60 * scale),
        "right_wrist": (hx + 0.50 * scale, hy - 0.60 * scale),
        "left_hip": (hx - 0.18 * scale, hy),
        "right_hip": (hx + 0.18 * scale, hy),
        "left_knee": (hx - 0.16 * scale + swing * 0.5, hy + 0.46 * scale - lift * 0.4),
        "right_knee": (hx + 0.16 * scale - swing * 0.5, hy + 0.46 * scale),
        "left_ankle": (hx - 0.14 * scale + swing, hy + 0.94 * scale - lift),
        "right_ankle": (hx + 0.14 * scale - swing * 0.6, hy + 0.94 * scale),
    }


class SyntheticScene:
    """Generates frames plus perfectly-known detections for a list of shots."""

    def __init__(
        self,
        scripts: tuple[ShotScript, ...] = DEFAULT_SCRIPTS,
        *,
        calibration: GoalCalibration | None = None,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        idle_frames: int = 8,
        seed: int = 7,
    ) -> None:
        self.scripts = scripts
        self.calibration = calibration or default_calibration()
        self.width = width
        self.height = height
        self.idle_frames = idle_frames
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    @property
    def expected_verdicts(self) -> list[str]:
        """What the engine should conclude, for the test suite to assert on."""
        mapping = {
            "goal": "GOAL",
            "saved": "SAVED",
            "wide": "OFF_TARGET",
            "over": "OFF_TARGET",
            "woodwork": "WOODWORK",
        }
        return [mapping[script.outcome] for script in self.scripts]

    def _keeper_position(self, script: ShotScript) -> Point:
        """Keeper stands near the middle, drifting toward the shot for a save."""
        if script.outcome == "saved":
            goal_x = script.target_goal_m[0]
        else:
            goal_x = self.calibration.width_m / 2.0
        return self.calibration.to_image((goal_x, 0.95))

    def _flight_path(self, script: ShotScript) -> list[Point]:
        """Image-space path from the strike point to the resolution point."""
        target_image = self.calibration.to_image(script.target_goal_m)
        start = script.start

        if script.outcome == "saved":
            # Stops at the keeper and comes back out.
            contact = _lerp(start, target_image, 0.93)
            forward = [
                _lerp(start, contact, (i + 1) / script.flight_frames)
                for i in range(script.flight_frames)
            ]
            # A parried ball comes back slower than it arrived.
            rebound_to = _lerp(contact, start, 0.30)
            back = [_lerp(contact, rebound_to, (i + 1) / 14) for i in range(14)]
            tail = [rebound_to] * script.settle_frames
            return forward + back + tail

        if script.outcome == "woodwork":
            contact = target_image
            forward = [
                _lerp(start, contact, (i + 1) / script.flight_frames)
                for i in range(script.flight_frames)
            ]
            rebound_to = _lerp(contact, start, 0.55)
            back = [_lerp(contact, rebound_to, (i + 1) / 8) for i in range(8)]
            return forward + back + [rebound_to] * script.settle_frames

        # An on-target shot is stopped by the net at the aim point; a miss
        # carries on past the goal before it comes down.
        end = target_image if script.outcome == "goal" else _lerp(start, target_image, 1.06)
        forward = [
            _lerp(start, end, (i + 1) / script.flight_frames)
            for i in range(script.flight_frames)
        ]
        settle = [end] * script.settle_frames
        return forward + settle

    def _arc(self, path: list[Point], script: ShotScript) -> list[Point]:
        """Add a little vertical arc so the trace does not look like a ruler."""
        if len(path) < 3:
            return path
        # Keep the arc small on placed finishes so the aim point stays honest.
        peak = 14.0 if script.outcome in {"over", "wide"} else 4.0
        out: list[Point] = []
        for index, point in enumerate(path):
            t = index / max(1, len(path) - 1)
            lift = peak * math.sin(math.pi * min(1.0, t * 1.15))
            out.append((point[0], point[1] - lift))
        return out

    # ------------------------------------------------------------------

    def frames(self) -> Iterator[SceneFrame]:
        frame_index = 0

        for script in self.scripts:
            keeper = self._keeper_position(script)
            path = self._arc(self._flight_path(script), script)
            ball_start = script.start

            # Approach: the striker walks in and the ball sits still.
            run_from = (ball_start[0] - 150.0, ball_start[1] + 74.0)
            for step in range(script.approach_frames):
                t = step / max(1, script.approach_frames - 1)
                hip = _lerp(run_from, (ball_start[0] - 26.0, ball_start[1] - 62.0), t)
                keypoints = _stick_figure(hip, scale=64.0, stride=step * 0.55)
                yield self._build(
                    frame_index,
                    ball=ball_start,
                    keypoints=keypoints,
                    keeper=keeper,
                    caption=f"run-up: {script.label}",
                )
                frame_index += 1

            # Strike and flight. The striker's planted foot stays put.
            planted_hip = (ball_start[0] - 26.0, ball_start[1] - 62.0)
            for step, point in enumerate(path):
                keypoints = _stick_figure(
                    planted_hip, scale=64.0, stride=1.9 + min(step, 6) * 0.12
                )
                yield self._build(
                    frame_index,
                    ball=point,
                    keypoints=keypoints,
                    keeper=keeper,
                    caption=f"flight: {script.label}",
                )
                frame_index += 1

            for _ in range(self.idle_frames):
                keypoints = _stick_figure(planted_hip, scale=64.0, stride=0.0)
                yield self._build(
                    frame_index,
                    ball=path[-1],
                    keypoints=keypoints,
                    keeper=keeper,
                    caption="",
                )
                frame_index += 1

    def _build(
        self,
        index: int,
        *,
        ball: Point,
        keypoints: dict[str, Point],
        keeper: Point,
        caption: str,
    ) -> SceneFrame:
        jitter = self.rng.normal(0.0, 0.6, size=2)
        ball_xy = (float(ball[0] + jitter[0]), float(ball[1] + jitter[1]))

        # Balls further away look smaller; a linear ramp on y is plenty here.
        depth = max(0.25, min(1.0, (ball_xy[1] - 150.0) / 520.0))
        radius = 5.0 + 7.0 * depth

        ball_box = Box(
            x1=ball_xy[0] - radius,
            y1=ball_xy[1] - radius,
            x2=ball_xy[0] + radius,
            y2=ball_xy[1] + radius,
            confidence=0.93,
            label="ball",
        )

        xs = [p[0] for p in keypoints.values()]
        ys = [p[1] for p in keypoints.values()]
        player_box = Box(
            x1=min(xs) - 8,
            y1=min(ys) - 10,
            x2=max(xs) + 8,
            y2=max(ys) + 6,
            confidence=0.95,
            label="player",
        )
        keeper_box = Box(
            x1=keeper[0] - 26,
            y1=keeper[1] - 74,
            x2=keeper[0] + 26,
            y2=keeper[1] + 10,
            confidence=0.9,
            label="goalkeeper",
        )

        pose = Pose(
            keypoints={name: (p[0], p[1], 0.96) for name, p in keypoints.items()},
            box=player_box,
        )

        detections = FrameDetections(
            frame_index=index,
            ball=ball_box,
            players=[player_box],
            goalkeepers=[keeper_box],
            poses=[pose],
        )
        return SceneFrame(
            index=index,
            detections=detections,
            player_keypoints=keypoints,
            keeper_centre=keeper,
            ball_radius_px=radius,
            caption=caption,
        )
