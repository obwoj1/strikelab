from __future__ import annotations

import pytest

from strikelab.constants import GOAL_WIDTH_M
from strikelab.geometry import CalibrationError
from strikelab.physics import (
    GroundCalibration,
    mps_to_kmh,
    mps_to_mph,
    shot_angle_deg,
    shot_quality,
)


class TestUnits:
    def test_conversions(self) -> None:
        assert mps_to_kmh(10.0) == pytest.approx(36.0)
        assert mps_to_mph(10.0) == pytest.approx(22.369, abs=1e-3)


class TestShotAngle:
    def test_penalty_spot_sees_a_wide_goal(self) -> None:
        # 11 m out, dead centre: 2 * atan(3.66 / 11) = 36.8 degrees.
        assert shot_angle_deg((0.0, 11.0)) == pytest.approx(36.83, abs=0.05)

    def test_angle_shrinks_with_distance(self) -> None:
        near = shot_angle_deg((0.0, 6.0))
        far = shot_angle_deg((0.0, 30.0))
        assert near > far

    def test_tight_angle_from_the_byline(self) -> None:
        wide = shot_angle_deg((20.0, 1.0))
        central = shot_angle_deg((0.0, 1.0))
        assert wide < 12.0 < central

    def test_on_the_goal_line_is_zero(self) -> None:
        assert shot_angle_deg((5.0, 0.0)) == 0.0

    def test_six_yard_box_centre(self) -> None:
        assert shot_angle_deg((0.0, 5.5)) == pytest.approx(67.0, abs=1.0)


class TestGroundCalibration:
    # A synthetic overhead-ish view of the six-yard box.
    IMAGE = [(200.0, 500.0), (800.0, 500.0), (740.0, 380.0), (260.0, 380.0)]

    def test_preset_maps_corners_to_pitch_coordinates(self) -> None:
        ground = GroundCalibration.from_preset("six-yard-box", self.IMAGE)
        assert ground.to_pitch(self.IMAGE[0]) == pytest.approx((-9.16, 0.0), abs=1e-6)
        assert ground.to_pitch(self.IMAGE[2]) == pytest.approx((9.16, 5.5), abs=1e-6)

    def test_distance_to_goal(self) -> None:
        ground = GroundCalibration.from_preset("six-yard-box", self.IMAGE)
        # Midpoint of the far edge of the six-yard box is 5.5 m straight out.
        midpoint = ((self.IMAGE[2][0] + self.IMAGE[3][0]) / 2, self.IMAGE[2][1])
        assert ground.distance_to_goal_m(midpoint) == pytest.approx(5.5, abs=1e-6)

    def test_unknown_preset_lists_the_options(self) -> None:
        with pytest.raises(CalibrationError, match="six-yard-box"):
            GroundCalibration.from_preset("halfway-line", self.IMAGE)

    def test_wrong_point_count_is_rejected(self) -> None:
        with pytest.raises(CalibrationError):
            GroundCalibration.from_points(self.IMAGE[:3], [(0.0, 0.0)] * 3)


class TestShotQuality:
    def test_close_central_corner_finish_beats_a_long_shot(self) -> None:
        tap_in = shot_quality(
            distance_m=5.0, angle_deg=60.0, zone="bottom-left", speed_mps=18.0, on_target=True
        )
        screamer = shot_quality(
            distance_m=30.0, angle_deg=15.0, zone="top-right", speed_mps=30.0, on_target=True
        )
        assert tap_in.score > screamer.score

    def test_corner_placement_beats_the_middle(self) -> None:
        corner = shot_quality(
            distance_m=12.0, angle_deg=35.0, zone="bottom-left", speed_mps=20.0, on_target=True
        )
        middle = shot_quality(
            distance_m=12.0, angle_deg=35.0, zone="middle-centre", speed_mps=20.0, on_target=True
        )
        assert corner.score > middle.score

    def test_angle_matters_at_a_fixed_distance(self) -> None:
        central = shot_quality(
            distance_m=12.0, angle_deg=50.0, zone="bottom-left", speed_mps=20.0, on_target=True
        )
        tight = shot_quality(
            distance_m=12.0, angle_deg=8.0, zone="bottom-left", speed_mps=20.0, on_target=True
        )
        assert central.score > tight.score

    def test_score_stays_in_range(self) -> None:
        for distance in (0.5, 5.0, 20.0, 60.0):
            result = shot_quality(
                distance_m=distance,
                angle_deg=40.0,
                zone="middle-centre",
                speed_mps=20.0,
                on_target=True,
            )
            assert 0.0 < result.score < 1.0

    def test_missing_inputs_fall_back_to_a_declared_prior(self) -> None:
        result = shot_quality(
            distance_m=None, angle_deg=None, zone=None, speed_mps=None, on_target=False
        )
        assert 0.0 < result.score < 1.0
        assert result.inputs["distance_source"] == "prior(16m)"
        assert result.inputs["angle_source"] == "prior(25deg)"

    def test_off_target_is_penalised(self) -> None:
        on = shot_quality(
            distance_m=10.0, angle_deg=40.0, zone="middle-centre", speed_mps=20.0, on_target=True
        )
        off = shot_quality(
            distance_m=10.0, angle_deg=40.0, zone=None, speed_mps=20.0, on_target=False
        )
        assert off.score < on.score

    def test_json_shape_is_serialisable(self) -> None:
        payload = shot_quality(
            distance_m=11.0, angle_deg=37.0, zone="bottom-left", speed_mps=22.0, on_target=True
        ).to_json()
        assert set(payload) == {
            "score",
            "distance_term",
            "angle_term",
            "placement_term",
            "speed_term",
            "inputs",
        }
