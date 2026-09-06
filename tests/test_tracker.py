from __future__ import annotations

import pytest

from strikelab.config import TrackerConfig
from strikelab.engine.tracker import BallTracker
from strikelab.types import Box


def ball_at(x: float, y: float, size: float = 10.0, confidence: float = 0.9) -> Box:
    return Box(x - size / 2, y - size / 2, x + size / 2, y + size / 2, confidence, "ball")


def make_tracker(**overrides: object) -> BallTracker:
    config = TrackerConfig(**overrides)  # type: ignore[arg-type]
    # 100 px per metre, 30 fps, goal 732 px wide.
    return BallTracker(config, pixels_per_metre=100.0, fps=30.0, goal_width_px=732.0)


class TestBasics:
    def test_no_track_until_a_ball_arrives(self) -> None:
        tracker = make_tracker()
        assert tracker.update(0, None) is None
        assert tracker.last is None

    def test_first_detection_starts_a_track(self) -> None:
        tracker = make_tracker()
        sample = tracker.update(0, ball_at(100.0, 100.0))
        assert sample is not None
        assert sample.position == pytest.approx((100.0, 100.0))
        assert sample.speed_mps == 0.0

    def test_low_confidence_detections_are_ignored(self) -> None:
        tracker = make_tracker(min_confidence=0.5)
        assert tracker.update(0, ball_at(100.0, 100.0, confidence=0.2)) is None


class TestSpeed:
    def test_constant_motion_gives_the_expected_speed(self) -> None:
        # 10 px per frame at 100 px/m and 30 fps is exactly 3 m/s.
        tracker = make_tracker(position_alpha=1.0)
        for index in range(8):
            sample = tracker.update(index, ball_at(100.0 + index * 10.0, 200.0))
        assert sample is not None
        assert sample.speed_mps == pytest.approx(3.0, rel=1e-6)

    def test_a_stationary_ball_reads_zero(self) -> None:
        tracker = make_tracker(position_alpha=1.0)
        for index in range(6):
            sample = tracker.update(index, ball_at(300.0, 300.0))
        assert sample is not None
        assert sample.speed_mps == pytest.approx(0.0, abs=1e-9)

    def test_ball_diameter_uses_the_goal_scale(self) -> None:
        # 0.22 m at 100 px/m.
        assert make_tracker().ball_diameter_px() == pytest.approx(22.0)


class TestAssociation:
    def test_a_distant_stray_ball_is_rejected(self) -> None:
        tracker = make_tracker(position_alpha=1.0)
        for index in range(4):
            tracker.update(index, ball_at(100.0 + index * 5.0, 200.0))
        before = tracker.last
        assert before is not None

        # A second ball on the far touchline, well beyond the gate.
        sample = tracker.update(4, ball_at(1100.0, 650.0))
        assert sample is not None
        assert sample.interpolated  # coasted rather than jumping
        assert sample.position[0] < 200.0

    def test_a_nearby_detection_is_accepted(self) -> None:
        tracker = make_tracker(position_alpha=1.0)
        for index in range(4):
            tracker.update(index, ball_at(100.0 + index * 5.0, 200.0))
        sample = tracker.update(4, ball_at(122.0, 201.0))
        assert sample is not None
        assert not sample.interpolated


class TestGapHandling:
    def test_coasts_through_a_short_occlusion(self) -> None:
        tracker = make_tracker(position_alpha=1.0, max_gap_frames=5)
        for index in range(5):
            tracker.update(index, ball_at(100.0 + index * 10.0, 200.0))

        coasted = tracker.update(5, None)
        assert coasted is not None
        assert coasted.interpolated
        assert coasted.position[0] == pytest.approx(150.0, abs=1.0)

    def test_drops_the_track_after_a_long_gap(self) -> None:
        tracker = make_tracker(position_alpha=1.0, max_gap_frames=3)
        for index in range(4):
            tracker.update(index, ball_at(100.0 + index * 10.0, 200.0))
        for index in range(4, 9):
            tracker.update(index, None)
        assert tracker.last is None
        assert not tracker.active

    def test_recovers_after_the_track_is_dropped(self) -> None:
        tracker = make_tracker(position_alpha=1.0, max_gap_frames=2)
        tracker.update(0, ball_at(100.0, 200.0))
        for index in range(1, 5):
            tracker.update(index, None)
        sample = tracker.update(5, ball_at(900.0, 600.0))
        assert sample is not None
        assert sample.position == pytest.approx((900.0, 600.0))


class TestSmoothing:
    def test_smoothing_damps_a_single_bad_frame(self) -> None:
        smooth = make_tracker(position_alpha=0.5)
        raw = make_tracker(position_alpha=1.0)
        for index in range(4):
            smooth.update(index, ball_at(100.0, 200.0))
            raw.update(index, ball_at(100.0, 200.0))

        jitter = ball_at(140.0, 200.0)
        smoothed_sample = smooth.update(4, jitter)
        raw_sample = raw.update(4, jitter)
        assert smoothed_sample is not None and raw_sample is not None
        assert smoothed_sample.position[0] < raw_sample.position[0]
        assert smoothed_sample.raw_position[0] == pytest.approx(140.0)
