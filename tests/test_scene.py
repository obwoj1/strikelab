"""Integration test: the full engine against the scripted five-shot scene.

This is the closest thing to an end-to-end regression check that does not need
model weights. If a change to the tracker or the state machine alters a verdict,
this fails.
"""

from __future__ import annotations

import json

import pytest

from strikelab.config import Config
from strikelab.engine import ShotEngine
from strikelab.sources.synthetic import DEFAULT_SCRIPTS, ShotScript, SyntheticScene
from strikelab.stats import summarise


@pytest.fixture(scope="module")
def engine() -> ShotEngine:
    scene = SyntheticScene()
    instance = ShotEngine(scene.calibration, config=Config(fps=30.0))
    for frame in scene.frames():
        instance.process(frame.detections)
    instance.finish()
    return instance


class TestFullScene:
    def test_every_scripted_shot_is_detected(self, engine: ShotEngine) -> None:
        assert len(engine.shots) == len(DEFAULT_SCRIPTS)

    def test_verdicts_match_ground_truth(self, engine: ShotEngine) -> None:
        assert [s.verdict for s in engine.shots] == SyntheticScene().expected_verdicts

    def test_placement_matches_where_each_shot_was_aimed(self, engine: ShotEngine) -> None:
        by_id = {shot.shot_id: shot for shot in engine.shots}
        assert by_id[1].zone == "bottom-left"
        assert by_id[4].zone == "top-right"
        # Off-target shots have no zone.
        assert by_id[3].zone is None
        assert by_id[5].zone is None

    def test_shot_ids_are_sequential(self, engine: ShotEngine) -> None:
        assert [s.shot_id for s in engine.shots] == [1, 2, 3, 4, 5]

    def test_speeds_are_physically_plausible(self, engine: ShotEngine) -> None:
        for shot in engine.shots:
            kmh = shot.peak_speed_mps * 3.6
            assert 20.0 < kmh < 140.0, f"shot {shot.shot_id} reported {kmh:.0f} km/h"

    def test_every_shot_carries_a_quality_score(self, engine: ShotEngine) -> None:
        for shot in engine.shots:
            assert shot.quality is not None
            assert 0.0 < float(shot.quality["score"]) < 1.0  # type: ignore[arg-type]

    def test_on_target_flag_agrees_with_the_verdict(self, engine: ShotEngine) -> None:
        for shot in engine.shots:
            if shot.verdict in {"GOAL", "SAVED"}:
                assert shot.on_target
            if shot.verdict == "OFF_TARGET":
                assert not shot.on_target

    def test_every_shot_has_a_trace(self, engine: ShotEngine) -> None:
        for shot in engine.shots:
            assert len(shot.trace) > 3

    def test_shots_serialise_to_json(self, engine: ShotEngine) -> None:
        for shot in engine.shots:
            payload = json.loads(json.dumps(shot.to_json()))
            assert payload["verdict"] == shot.verdict


class TestSummary:
    def test_session_totals(self, engine: ShotEngine) -> None:
        summary = summarise(engine.shots)
        assert summary.shots == 5
        assert summary.goals == 2
        assert summary.saved == 1
        assert summary.off_target == 2
        assert summary.on_target == 3
        assert summary.conversion_rate == pytest.approx(0.4)
        assert summary.on_target_rate == pytest.approx(0.6)

    def test_summary_serialises(self, engine: ShotEngine) -> None:
        payload = json.loads(json.dumps(summarise(engine.shots).to_json()))
        assert payload["shots"] == 5

    def test_empty_session_is_safe(self) -> None:
        summary = summarise([])
        assert summary.shots == 0
        assert summary.conversion_rate == 0.0
        assert summary.mean_speed_kmh == 0.0


class TestSceneVariations:
    def test_a_scene_with_one_shot(self) -> None:
        scene = SyntheticScene((DEFAULT_SCRIPTS[0],))
        engine = ShotEngine(scene.calibration, config=Config(fps=30.0))
        for frame in scene.frames():
            engine.process(frame.detections)
        engine.finish()
        assert [s.verdict for s in engine.shots] == ["GOAL"]

    def test_woodwork_script(self) -> None:
        calibration = SyntheticScene().calibration
        # Aim 4 cm inside the left post.
        script = ShotScript(
            outcome="woodwork",
            start=(560.0, 630.0),
            target_goal_m=(0.04, 1.10),
            label="off the inside of the post",
        )
        scene = SyntheticScene((script,), calibration=calibration)
        engine = ShotEngine(scene.calibration, config=Config(fps=30.0))
        for frame in scene.frames():
            engine.process(frame.detections)
        engine.finish()
        assert [s.verdict for s in engine.shots] == ["WOODWORK"]
