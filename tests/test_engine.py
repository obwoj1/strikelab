"""End-to-end engine tests driven by scripted detections.

These build tiny synthetic shots by hand so each scenario isolates one decision
the state machine has to make. The full five-shot scene is covered separately in
test_scene.py.
"""

from __future__ import annotations

import pytest

from strikelab.config import Config, ShotConfig, TrackerConfig
from strikelab.engine import ShotEngine
from strikelab.geometry import GoalCalibration
from strikelab.types import Box, FrameDetections, Point, Pose

# 100 px per metre, goal mouth from (100,100) to (832,344).
GOAL = [(100.0, 100.0), (832.0, 100.0), (832.0, 344.0), (100.0, 344.0)]


def calibration() -> GoalCalibration:
    return GoalCalibration.from_corners(GOAL)


def config(**shot_overrides: object) -> Config:
    return Config(
        tracker=TrackerConfig(position_alpha=1.0),
        shot=ShotConfig(**shot_overrides),  # type: ignore[arg-type]
        fps=30.0,
    )


def ball(x: float, y: float) -> Box:
    return Box(x - 5, y - 5, x + 5, y + 5, 0.95, "ball")


def pose_with_foot(at: Point) -> Pose:
    return Pose(keypoints={"right_ankle": (at[0], at[1], 0.9)})


def keeper_at(x: float, y: float) -> Box:
    return Box(x - 25, y - 70, x + 25, y + 5, 0.9, "goalkeeper")


class ShotBuilder:
    """Assembles a frame sequence: a still ball, a contact, then a flight path."""

    def __init__(self) -> None:
        self.frames: list[FrameDetections] = []
        self._index = 0

    def idle(self, at: Point, frames: int = 6, foot: Point | None = None) -> "ShotBuilder":
        for _ in range(frames):
            poses = [pose_with_foot(foot)] if foot else []
            self.frames.append(
                FrameDetections(frame_index=self._index, ball=ball(*at), poses=poses)
            )
            self._index += 1
        return self

    def fly(
        self,
        path: list[Point],
        *,
        keeper: Point | None = None,
    ) -> "ShotBuilder":
        for point in path:
            self.frames.append(
                FrameDetections(
                    frame_index=self._index,
                    ball=ball(*point),
                    goalkeepers=[keeper_at(*keeper)] if keeper else [],
                )
            )
            self._index += 1
        return self

    def rest(self, at: Point, frames: int = 10, keeper: Point | None = None) -> "ShotBuilder":
        return self.fly([at] * frames, keeper=keeper)


def straight(start: Point, end: Point, steps: int) -> list[Point]:
    return [
        (
            start[0] + (end[0] - start[0]) * (i + 1) / steps,
            start[1] + (end[1] - start[1]) * (i + 1) / steps,
        )
        for i in range(steps)
    ]


def run(frames: list[FrameDetections], cfg: Config | None = None) -> ShotEngine:
    engine = ShotEngine(calibration(), config=cfg or config())
    for frame in frames:
        engine.process(frame)
    engine.finish()
    return engine


class TestShotDetection:
    def test_a_struck_ball_becomes_a_shot(self) -> None:
        start = (466.0, 620.0)
        builder = ShotBuilder().idle(start, foot=(start[0] - 14, start[1] + 4))
        builder.fly(straight(start, (466.0, 222.0), 10)).rest((466.0, 222.0))
        engine = run(builder.frames)
        assert len(engine.shots) == 1

    def test_a_slow_pass_is_not_a_shot(self) -> None:
        start = (466.0, 620.0)
        builder = ShotBuilder().idle(start, foot=(start[0] - 14, start[1] + 4))
        # 4 px/frame at 100 px/m and 30 fps is 1.2 m/s: a gentle roll.
        builder.fly(straight(start, (466.0, 580.0), 10)).rest((466.0, 580.0))
        engine = run(builder.frames)
        assert engine.shots == []

    def test_a_fast_ball_with_no_contact_is_not_a_shot(self) -> None:
        start = (466.0, 620.0)
        builder = ShotBuilder().idle(start, frames=6)  # no pose, no player box
        builder.fly(straight(start, (466.0, 222.0), 10)).rest((466.0, 222.0))
        engine = run(builder.frames)
        assert engine.shots == []

    def test_a_ball_struck_away_from_goal_is_not_a_shot(self) -> None:
        start = (466.0, 500.0)
        builder = ShotBuilder().idle(start, foot=(start[0] - 14, start[1] + 4))
        builder.fly(straight(start, (466.0, 700.0), 10)).rest((466.0, 700.0))
        engine = run(builder.frames)
        assert engine.shots == []

    def test_the_contact_keypoint_is_recorded(self) -> None:
        start = (466.0, 620.0)
        builder = ShotBuilder().idle(start, foot=(start[0] - 14, start[1] + 4))
        builder.fly(straight(start, (466.0, 222.0), 10)).rest((466.0, 222.0))
        engine = run(builder.frames)
        assert engine.shots[0].contact_keypoint == "right_ankle"


class TestVerdicts:
    def _shoot(self, target: Point, *, keeper: Point | None = None) -> ShotEngine:
        start = (466.0, 620.0)
        builder = ShotBuilder().idle(start, foot=(start[0] - 14, start[1] + 4))
        builder.fly(straight(start, target, 10), keeper=keeper)
        builder.rest(target, frames=12, keeper=keeper)
        return run(builder.frames)

    def test_into_the_net_is_a_goal(self) -> None:
        engine = self._shoot((300.0, 300.0))  # low and left, inside the mouth
        shot = engine.shots[0]
        assert shot.verdict == "GOAL"
        assert shot.on_target
        assert shot.zone == "bottom-left"

    def test_wide_of_the_post_is_off_target(self) -> None:
        engine = self._shoot((40.0, 250.0))  # left of the left post
        shot = engine.shots[0]
        assert shot.verdict == "OFF_TARGET"
        assert not shot.on_target
        assert shot.zone is None

    def test_over_the_bar_is_off_target(self) -> None:
        # Passes visually across the mouth on the way up, then ends above it.
        engine = self._shoot((466.0, 40.0))
        assert engine.shots[0].verdict == "OFF_TARGET"

    def test_a_ball_stopped_at_the_keeper_is_a_save(self) -> None:
        start = (466.0, 620.0)
        contact = (466.0, 250.0)
        builder = ShotBuilder().idle(start, foot=(start[0] - 14, start[1] + 4))
        builder.fly(straight(start, contact, 10), keeper=contact)
        # Parried straight back out.
        builder.fly(straight(contact, (466.0, 430.0), 8), keeper=contact)
        builder.rest((466.0, 430.0), frames=10, keeper=contact)
        engine = run(builder.frames)
        assert engine.shots[0].verdict == "SAVED"

    def test_a_rebound_off_the_post_is_woodwork(self) -> None:
        start = (466.0, 620.0)
        post = (104.0, 250.0)  # 0.04 m inside the left post
        builder = ShotBuilder().idle(start, foot=(start[0] - 14, start[1] + 4))
        builder.fly(straight(start, post, 10))
        builder.fly(straight(post, (300.0, 500.0), 8))
        builder.rest((300.0, 500.0), frames=10)
        engine = run(builder.frames)
        assert engine.shots[0].verdict == "WOODWORK"

    def test_a_deflection_short_of_the_goal_is_blocked(self) -> None:
        start = (466.0, 620.0)
        block = (466.0, 470.0)
        builder = ShotBuilder().idle(start, foot=(start[0] - 14, start[1] + 4))
        builder.fly(straight(start, block, 4))
        builder.fly(straight(block, (466.0, 660.0), 8))
        builder.rest((466.0, 660.0), frames=10)
        engine = run(builder.frames)
        shot = engine.shots[0]
        assert shot.verdict in {"BLOCKED", "SAVED"}
        assert not shot.on_target


class TestMetrics:
    def _goal(self) -> ShotEngine:
        start = (466.0, 620.0)
        builder = ShotBuilder().idle(start, foot=(start[0] - 14, start[1] + 4))
        builder.fly(straight(start, (300.0, 300.0), 10)).rest((300.0, 300.0), frames=12)
        return run(builder.frames)

    def test_speed_is_reported_in_a_plausible_range(self) -> None:
        shot = self._goal().shots[0]
        # ~36 px/frame at 100 px/m and 30 fps is ~10.8 m/s.
        assert 8.0 < shot.peak_speed_mps < 14.0

    def test_quality_is_attached(self) -> None:
        shot = self._goal().shots[0]
        assert shot.quality is not None
        assert 0.0 < float(shot.quality["score"]) < 1.0  # type: ignore[arg-type]

    def test_duration_is_positive(self) -> None:
        shot = self._goal().shots[0]
        assert shot.duration_s is not None and shot.duration_s > 0

    def test_distance_stays_none_without_ground_calibration(self) -> None:
        shot = self._goal().shots[0]
        assert shot.distance_m is None
        assert shot.quality is not None
        assert shot.quality["inputs"]["distance_source"] == "prior(16m)"  # type: ignore[index]

    def test_json_round_trips(self) -> None:
        payload = self._goal().shots[0].to_json()
        assert payload["verdict"] == "GOAL"
        assert payload["release"]["speed_kmh"] > 0  # type: ignore[index]


class TestMultipleShots:
    def test_two_shots_are_numbered_independently(self) -> None:
        start = (466.0, 620.0)
        foot = (start[0] - 14, start[1] + 4)
        builder = ShotBuilder().idle(start, foot=foot)
        builder.fly(straight(start, (300.0, 300.0), 10)).rest((300.0, 300.0), frames=14)
        # The ball is placed back on the spot. The tracker deliberately refuses
        # to teleport onto it, so it drops the stale track first and re-acquires;
        # allow enough idle frames for that to happen.
        builder.idle(start, frames=26, foot=foot)
        builder.fly(straight(start, (700.0, 300.0), 10)).rest((700.0, 300.0), frames=14)
        engine = run(builder.frames)
        assert [s.shot_id for s in engine.shots] == [1, 2]
        assert all(s.verdict == "GOAL" for s in engine.shots)


class TestFallbackContact:
    def test_a_player_box_can_arm_the_machine_without_a_pose_model(self) -> None:
        start = (466.0, 620.0)
        frames: list[FrameDetections] = []
        player = Box(start[0] - 30, start[1] - 120, start[0] + 10, start[1] + 6, 0.9, "player")
        for index in range(6):
            frames.append(
                FrameDetections(frame_index=index, ball=ball(*start), players=[player])
            )
        for offset, point in enumerate(straight(start, (300.0, 300.0), 10)):
            frames.append(FrameDetections(frame_index=6 + offset, ball=ball(*point)))
        for offset in range(12):
            frames.append(FrameDetections(frame_index=16 + offset, ball=ball(300.0, 300.0)))

        engine = run(frames)
        assert len(engine.shots) == 1
        assert engine.shots[0].contact_keypoint == "player_box_base"
