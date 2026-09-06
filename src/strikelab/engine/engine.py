"""ShotEngine: frame in, events out.

Deliberately decoupled from both the detector and the video source, exactly as
the basketball project's engine is. That is what lets the test suite drive the
whole thing with scripted detections and no model weights at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import Config
from ..geometry import GoalCalibration
from ..physics import GroundCalibration, shot_angle_deg, shot_quality
from ..types import CONTACT_KEYPOINTS, Box, FrameDetections, Point, Pose
from .state import FrameFacts, Shot, ShotStateMachine, State
from .tracker import BallSample, BallTracker


@dataclass
class FrameResult:
    """What the renderer needs to draw one frame."""

    frame_index: int
    ball: BallSample | None
    detections: FrameDetections
    state: str
    goal_point: Point | None
    inside_mouth: bool
    contact: tuple[str, Point] | None
    completed_shot: Shot | None = None
    active_trace: list[Point] = field(default_factory=list)


class ShotEngine:
    def __init__(
        self,
        calibration: GoalCalibration,
        *,
        config: Config | None = None,
        ground: GroundCalibration | None = None,
    ) -> None:
        self.config = config or Config()
        self.calibration = calibration
        self.ground = ground

        goal_width_px = math.hypot(
            calibration.corners_tl_tr_br_bl[1][0] - calibration.corners_tl_tr_br_bl[0][0],
            calibration.corners_tl_tr_br_bl[1][1] - calibration.corners_tl_tr_br_bl[0][1],
        )
        self.tracker = BallTracker(
            self.config.tracker,
            pixels_per_metre=calibration.pixels_per_metre(),
            fps=self.config.fps,
            goal_width_px=goal_width_px,
        )
        self.machine = ShotStateMachine(
            self.config.shot,
            calibration,
            ball_diameter_px=self.tracker.ball_diameter_px(),
        )
        self.shots: list[Shot] = []

    # ------------------------------------------------------------------

    def _find_contact(
        self, ball_position: Point, poses: list[Pose]
    ) -> tuple[str, Point] | None:
        """Nearest striking surface within the contact radius.

        Feet first, then head and knee. Unlike basketball's wrist test, which
        can rely on a clean single hand, we take the most confident of several
        candidate joints because ankles are so often occluded by the other leg.
        """
        radius = self.machine.contact_radius_px
        best: tuple[float, str, Point] | None = None

        for pose in poses:
            for name in CONTACT_KEYPOINTS:
                point = pose.point(name)
                if point is None:
                    continue
                distance = math.hypot(
                    ball_position[0] - point[0], ball_position[1] - point[1]
                )
                if distance <= radius and (best is None or distance < best[0]):
                    best = (distance, name, point)

        if best is not None:
            return (best[1], best[2])
        return None

    def _fallback_contact(
        self, ball_position: Point, people: list[Box]
    ) -> tuple[str, Point] | None:
        """When no pose model is available, use the bottom of a player's box.

        Less precise than an ankle keypoint, but enough to arm the state
        machine, and it keeps the engine usable with a detection-only setup.
        """
        radius = self.machine.contact_radius_px * 1.6
        best: tuple[float, Point] | None = None
        for person in people:
            foot = person.bottom_centre
            distance = math.hypot(
                ball_position[0] - foot[0], ball_position[1] - foot[1]
            )
            if distance <= radius and (best is None or distance < best[0]):
                best = (distance, foot)
        if best is not None:
            return ("player_box_base", best[1])
        return None

    # ------------------------------------------------------------------

    def process(self, detections: FrameDetections) -> FrameResult:
        sample = self.tracker.update(detections.frame_index, detections.ball)

        if sample is None:
            if self.machine.state is State.IN_FLIGHT:
                # The ball vanished mid-shot; let the machine decide whether
                # there is enough evidence to call it.
                completed = self.machine.force_resolve()
                if completed is not None:
                    self._finalise(completed)
                    return FrameResult(
                        frame_index=detections.frame_index,
                        ball=None,
                        detections=detections,
                        state=self.machine.state.value,
                        goal_point=None,
                        inside_mouth=False,
                        contact=None,
                        completed_shot=completed,
                    )
            return FrameResult(
                frame_index=detections.frame_index,
                ball=None,
                detections=detections,
                state=self.machine.state.value,
                goal_point=None,
                inside_mouth=False,
                contact=None,
            )

        goal_point: Point | None
        try:
            goal_point = self.calibration.to_goal_plane(sample.position)
        except Exception:  # noqa: BLE001 - a ball on the horizon line is not fatal
            goal_point = None

        inside = (
            self.calibration.inside_goal_plane(goal_point) if goal_point else False
        )
        distance_to_frame = (
            self.calibration.distance_to_frame_m(goal_point) if goal_point else None
        )

        contact = self._find_contact(sample.position, detections.poses)
        if contact is None and not detections.poses:
            contact = self._fallback_contact(sample.position, detections.all_people)

        facts = FrameFacts(
            frame_index=detections.frame_index,
            position=sample.position,
            velocity=sample.velocity,
            speed_mps=sample.speed_mps,
            goal_point=goal_point,
            inside_mouth=inside,
            distance_to_frame_m=distance_to_frame,
            contact=contact,
            keeper_positions=[k.centre for k in detections.goalkeepers],
            defender_positions=[p.centre for p in detections.players],
            interpolated=sample.interpolated,
        )

        completed = self.machine.update(facts)
        if completed is not None:
            self._finalise(completed)

        trace = [t.position for t in self.machine._trace]  # noqa: SLF001 - render aid

        return FrameResult(
            frame_index=detections.frame_index,
            ball=sample,
            detections=detections,
            state=self.machine.state.value,
            goal_point=goal_point,
            inside_mouth=inside,
            contact=contact,
            completed_shot=completed,
            active_trace=trace,
        )

    # ------------------------------------------------------------------

    def _finalise(self, shot: Shot) -> None:
        """Attach the metrics that need calibration the state machine lacks."""
        shot.duration_s = (shot.end_frame - shot.release_frame) / self.config.fps

        if self.ground is not None:
            try:
                pitch = self.ground.to_pitch(shot.release_position)
                shot.distance_m = float(math.hypot(pitch[0], pitch[1]))
                shot.angle_deg = shot_angle_deg(pitch, self.calibration.width_m)
            except Exception:  # noqa: BLE001 - fall back to the prior
                shot.distance_m = None
                shot.angle_deg = None
                shot.notes.append("ground calibration failed for this release point")

        speed = shot.peak_speed_mps if shot.peak_speed_mps > 0 else shot.release_speed_mps
        if speed < self.config.shot.min_reportable_speed_mps:
            shot.notes.append("release speed below the reportable floor")

        shot.quality = shot_quality(
            distance_m=shot.distance_m,
            angle_deg=shot.angle_deg,
            zone=shot.zone,
            speed_mps=speed if speed > 0 else None,
            on_target=shot.on_target,
        ).to_json()

        self.shots.append(shot)

    def finish(self) -> Shot | None:
        """Call once at end of stream to close out anything still in flight."""
        completed = self.machine.force_resolve()
        if completed is not None:
            self._finalise(completed)
        return completed
