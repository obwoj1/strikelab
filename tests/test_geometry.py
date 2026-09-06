from __future__ import annotations

import math

import pytest

from strikelab.constants import GOAL_HEIGHT_M, GOAL_WIDTH_M
from strikelab.geometry import CalibrationError, GoalCalibration, order_goal_corners

# A fronto-parallel goal: 732 px wide, 244 px tall, so exactly 100 px per metre.
SQUARE = [(100.0, 100.0), (832.0, 100.0), (832.0, 344.0), (100.0, 344.0)]

# A realistically skewed one, viewed from off to one side.
SKEWED = [(430.0, 196.0), (868.0, 188.0), (902.0, 372.0), (398.0, 380.0)]


def square() -> GoalCalibration:
    return GoalCalibration.from_corners(SQUARE)


class TestCornerOrdering:
    def test_orders_scrambled_corners(self) -> None:
        scrambled = [SQUARE[2], SQUARE[0], SQUARE[3], SQUARE[1]]
        ordered = order_goal_corners(scrambled)
        assert [tuple(p) for p in ordered] == [
            (100.0, 100.0),
            (832.0, 100.0),
            (832.0, 344.0),
            (100.0, 344.0),
        ]

    def test_rejects_wrong_number_of_corners(self) -> None:
        with pytest.raises(CalibrationError):
            GoalCalibration.from_corners(SQUARE[:3])

    def test_rejects_non_finite_corners(self) -> None:
        with pytest.raises(CalibrationError):
            GoalCalibration.from_corners([(0.0, 0.0), (1.0, float("nan")), (2.0, 2.0), (0.0, 2.0)])

    def test_rejects_collinear_corners(self) -> None:
        with pytest.raises(CalibrationError):
            GoalCalibration.from_corners(
                [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
            )


class TestGoalPlaneMapping:
    def test_corners_map_to_the_goal_rectangle(self) -> None:
        calibration = square()
        assert calibration.to_goal_plane((100.0, 344.0)) == pytest.approx((0.0, 0.0), abs=1e-6)
        assert calibration.to_goal_plane((832.0, 344.0)) == pytest.approx(
            (GOAL_WIDTH_M, 0.0), abs=1e-6
        )
        assert calibration.to_goal_plane((100.0, 100.0)) == pytest.approx(
            (0.0, GOAL_HEIGHT_M), abs=1e-6
        )

    def test_centre_of_the_mouth(self) -> None:
        x, y = square().to_goal_plane((466.0, 222.0))
        assert x == pytest.approx(GOAL_WIDTH_M / 2, abs=1e-6)
        assert y == pytest.approx(GOAL_HEIGHT_M / 2, abs=1e-6)

    def test_round_trips_through_image_space(self) -> None:
        calibration = GoalCalibration.from_corners(SKEWED)
        for goal_point in [(0.5, 0.5), (3.66, 1.22), (7.0, 2.2)]:
            image = calibration.to_image(goal_point)
            assert calibration.to_goal_plane(image) == pytest.approx(goal_point, abs=1e-6)

    def test_skewed_view_still_maps_corners_exactly(self) -> None:
        calibration = GoalCalibration.from_corners(SKEWED)
        assert calibration.to_goal_plane(SKEWED[3]) == pytest.approx((0.0, 0.0), abs=1e-6)
        assert calibration.to_goal_plane(SKEWED[1]) == pytest.approx(
            (GOAL_WIDTH_M, GOAL_HEIGHT_M), abs=1e-6
        )

    def test_pixels_per_metre(self) -> None:
        assert square().pixels_per_metre() == pytest.approx(100.0, rel=1e-9)


class TestContainment:
    def test_inside_and_outside(self) -> None:
        calibration = square()
        assert calibration.contains((466.0, 222.0))
        assert not calibration.contains((50.0, 222.0))  # wide left
        assert not calibration.contains((466.0, 50.0))  # over the bar

    def test_margin_lets_near_misses_count(self) -> None:
        calibration = square()
        just_wide = (90.0, 222.0)  # 0.1 m outside the left post
        assert not calibration.contains(just_wide)
        assert calibration.contains(just_wide, margin_m=0.2)


class TestDistanceToFrame:
    def test_positive_inside_negative_outside(self) -> None:
        calibration = square()
        assert calibration.distance_to_frame_m((3.66, 1.22)) == pytest.approx(1.22)
        assert calibration.distance_to_frame_m((0.1, 1.22)) == pytest.approx(0.1)
        assert calibration.distance_to_frame_m((-0.5, 1.22)) == pytest.approx(-0.5)

    def test_diagonal_miss_uses_euclidean_distance(self) -> None:
        calibration = square()
        result = calibration.distance_to_frame_m((-3.0, GOAL_HEIGHT_M + 4.0))
        assert result == pytest.approx(-5.0)


class TestZones:
    @pytest.mark.parametrize(
        ("goal_point", "expected"),
        [
            ((0.4, 0.3), "bottom-left"),
            ((3.66, 0.3), "bottom-centre"),
            ((7.0, 0.3), "bottom-right"),
            ((0.4, 1.22), "middle-left"),
            ((3.66, 1.22), "middle-centre"),
            ((0.4, 2.2), "top-left"),
            ((7.0, 2.2), "top-right"),
        ],
    )
    def test_zone_lookup(self, goal_point: tuple[float, float], expected: str) -> None:
        assert square().zone_of(goal_point) == expected

    def test_no_zone_for_an_off_target_shot(self) -> None:
        assert square().zone_of((-1.0, 1.0)) is None

    def test_corner_of_the_mouth_stays_in_range(self) -> None:
        assert square().zone_of((GOAL_WIDTH_M, GOAL_HEIGHT_M)) == "top-right"

    def test_grid_has_four_segments(self) -> None:
        assert len(square().zone_grid_segments()) == 4


class TestMissDescription:
    def test_over_the_bar(self) -> None:
        assert "over" in square().miss_description((3.66, 3.44))

    def test_wide_left_and_right(self) -> None:
        calibration = square()
        assert "wide left" in calibration.miss_description((-1.5, 1.0))
        assert "wide right" in calibration.miss_description((GOAL_WIDTH_M + 2.0, 1.0))

    def test_over_and_wide_together(self) -> None:
        description = square().miss_description((-1.0, 4.0))
        assert "over" in description and "wide left" in description
