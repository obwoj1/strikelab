"""The shot state machine.

Basketball's release trigger is "the ball separated from the wrist by 0.35 m
while moving upward, for 3 consecutive frames". Neither half of that transfers:

  * a soccer ball is struck with the foot, and ankle keypoints are noisier and
    far more often occluded than wrists,
  * a soccer ball starts on the ground and a driven shot can stay under a metre
    the whole way, so "moving upward" would reject most real shots.

So the trigger here is *sudden acceleration away from a foot, toward the goal*:
a contact arms the machine, then a sustained speed spike with a goalward
component fires it. That also naturally rejects the things that would otherwise
look like shots - passes (no speed spike relative to the pass), dribbles (no
goalward sustain), and clearances (no goalward component).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from ..config import ShotConfig
from ..geometry import GoalCalibration, cosine
from ..types import Point, Verdict


class State(Enum):
    IDLE = "idle"
    ARMED = "armed"
    IN_FLIGHT = "in_flight"


@dataclass
class FrameFacts:
    """Everything the state machine needs about one frame."""

    frame_index: int
    position: Point
    velocity: Point
    speed_mps: float
    goal_point: Point | None
    inside_mouth: bool
    distance_to_frame_m: float | None
    contact: tuple[str, Point] | None  # (keypoint name, position)
    keeper_positions: Sequence[Point] = ()
    defender_positions: Sequence[Point] = ()
    interpolated: bool = False


@dataclass
class ShotTrace:
    """The per-frame record kept for one shot, for rendering and review."""

    frame_index: int
    position: Point
    goal_point: Point | None
    speed_mps: float


@dataclass
class Shot:
    """One completed attempt."""

    shot_id: int
    verdict: Verdict
    confidence: str
    start_frame: int
    release_frame: int
    end_frame: int
    release_position: Point
    release_speed_mps: float
    peak_speed_mps: float
    contact_keypoint: str | None
    entry_goal_point: Point | None
    entry_frame: int | None
    zone: str | None
    on_target: bool
    distance_to_frame_m: float | None
    trace: list[ShotTrace] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # Filled in by the engine, which owns the optional ground calibration.
    distance_m: float | None = None
    angle_deg: float | None = None
    quality: dict[str, object] | None = None
    duration_s: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "shot_id": self.shot_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "on_target": self.on_target,
            "zone": self.zone,
            "release": {
                "frame": self.release_frame,
                "position_px": [round(v, 1) for v in self.release_position],
                "speed_mps": round(self.release_speed_mps, 2),
                "speed_kmh": round(self.release_speed_mps * 3.6, 1),
                "contact_keypoint": self.contact_keypoint,
            },
            "peak_speed_mps": round(self.peak_speed_mps, 2),
            "peak_speed_kmh": round(self.peak_speed_mps * 3.6, 1),
            "entry": {
                "frame": self.entry_frame,
                "goal_plane_m": (
                    [round(v, 3) for v in self.entry_goal_point]
                    if self.entry_goal_point
                    else None
                ),
                "distance_to_frame_m": (
                    round(self.distance_to_frame_m, 3)
                    if self.distance_to_frame_m is not None
                    else None
                ),
            },
            "distance_m": round(self.distance_m, 2) if self.distance_m is not None else None,
            "angle_deg": round(self.angle_deg, 1) if self.angle_deg is not None else None,
            "duration_s": round(self.duration_s, 3) if self.duration_s is not None else None,
            "quality": self.quality,
            "frames": {"start": self.start_frame, "end": self.end_frame},
            "notes": self.notes,
        }


class ShotStateMachine:
    """Consumes FrameFacts, emits Shot objects."""

    def __init__(
        self,
        config: ShotConfig,
        calibration: GoalCalibration,
        *,
        ball_diameter_px: float,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.contact_radius_px = ball_diameter_px * config.contact_radius_ball_diameters

        self.state = State.IDLE
        self._shot_counter = 0

        self._last_contact: tuple[int, str, Point] | None = None
        self._pre_contact_speed = 0.0
        self._release_streak = 0
        self._stop_streak = 0

        self._current: dict[str, object] = {}
        self._trace: list[ShotTrace] = []
        self._speed_history: list[float] = []

    # ------------------------------------------------------------------
    # helpers

    def _goalward(self, facts: FrameFacts) -> float:
        """Cosine between the ball's velocity and the direction to the goal.

        Only valid *before* the ball reaches the goal: once it passes the goal
        centre this bearing flips, which is why the rebound test below uses the
        fixed release direction instead.
        """
        target = self.calibration.to_image(
            (self.calibration.width_m / 2.0, self.calibration.height_m / 2.0)
        )
        to_goal = (target[0] - facts.position[0], target[1] - facts.position[1])
        return cosine(facts.velocity, to_goal)

    def _along_shot(self, facts: FrameFacts) -> float:
        """Cosine between the ball's velocity and the direction it was struck.

        Fixed at release, so it stays meaningful after the ball has passed the
        goal. A strong negative here is a genuine reversal: a save, a block, or
        the ball coming back off the frame.
        """
        direction = self._current.get("release_direction")
        if not direction:
            return 0.0
        return cosine(facts.velocity, direction)  # type: ignore[arg-type]

    def _penetration(self, facts: FrameFacts) -> float:
        """How far along the shot direction the ball has travelled, in pixels.

        The maximum of this over the flight is the moment of deepest
        penetration - the ball in the net, in the keeper's hands, past the post,
        or above the bar. That frame, not the most goal-central frame, is what
        decides the verdict.
        """
        direction = self._current.get("release_direction")
        origin = self._current.get("release_position")
        if not direction or not origin:
            return 0.0
        dx = facts.position[0] - origin[0]  # type: ignore[index]
        dy = facts.position[1] - origin[1]  # type: ignore[index]
        return dx * direction[0] + dy * direction[1]  # type: ignore[index]

    def _nearest(self, point: Point, candidates: Sequence[Point]) -> float | None:
        if not candidates:
            return None
        return min(math.hypot(point[0] - c[0], point[1] - c[1]) for c in candidates)

    # ------------------------------------------------------------------
    # main entry point

    def update(self, facts: FrameFacts) -> Shot | None:
        """Advance one frame. Returns a Shot on the frame a shot resolves."""
        if facts.contact is not None:
            self._last_contact = (facts.frame_index, facts.contact[0], facts.contact[1])
            if self.state is State.IDLE:
                self._pre_contact_speed = max(self._pre_contact_speed, facts.speed_mps)
                self.state = State.ARMED

        if self.state is State.IDLE:
            # Track a rolling baseline so we know what "sped up" means.
            self._pre_contact_speed = 0.7 * self._pre_contact_speed + 0.3 * facts.speed_mps
            return None

        if self.state is State.ARMED:
            self._check_release(facts)
            return None

        return self._advance_flight(facts)

    # ------------------------------------------------------------------
    # ARMED -> IN_FLIGHT

    def _check_release(self, facts: FrameFacts) -> None:
        contact_age = (
            facts.frame_index - self._last_contact[0] if self._last_contact else 10**6
        )
        if contact_age > self.config.contact_memory_frames:
            self.state = State.IDLE
            self._release_streak = 0
            return

        baseline = max(self._pre_contact_speed, 0.5)
        fast_enough = facts.speed_mps >= self.config.release_speed_mps
        accelerated = facts.speed_mps >= baseline * self.config.release_accel_ratio
        goalward = self._goalward(facts) >= self.config.min_goalward_cosine

        if fast_enough and accelerated and goalward:
            self._release_streak += 1
        else:
            self._release_streak = 0

        if self._release_streak < self.config.release_confirm_frames:
            return

        # Fire. Back-date the release to the first frame of the streak so the
        # reported release speed is the strike, not three frames of drag later.
        self._shot_counter += 1
        release_frame = facts.frame_index - self.config.release_confirm_frames + 1
        speed = math.hypot(*facts.velocity)
        direction = (
            (facts.velocity[0] / speed, facts.velocity[1] / speed)
            if speed > 1e-9
            else (0.0, -1.0)
        )
        self._current = {
            "shot_id": self._shot_counter,
            "start_frame": self._last_contact[0] if self._last_contact else release_frame,
            "release_frame": release_frame,
            "release_position": facts.position,
            "release_direction": direction,
            "release_speed_mps": facts.speed_mps,
            "contact_keypoint": self._last_contact[1] if self._last_contact else None,
            "best_entry": facts.goal_point,
            "best_entry_frame": facts.frame_index,
            "best_distance": facts.distance_to_frame_m,
            "max_penetration": 0.0,
            "crossed_inside": facts.inside_mouth,
            "notes": [],
        }
        self._trace = []
        self._speed_history = []
        self._stop_streak = 0
        self._release_streak = 0
        self.state = State.IN_FLIGHT

    # ------------------------------------------------------------------
    # IN_FLIGHT -> resolved

    def _advance_flight(self, facts: FrameFacts) -> Shot | None:
        self._trace.append(
            ShotTrace(
                frame_index=facts.frame_index,
                position=facts.position,
                goal_point=facts.goal_point,
                speed_mps=facts.speed_mps,
            )
        )
        self._speed_history.append(facts.speed_mps)

        # Record the frame of deepest penetration along the shot direction.
        # This is the soccer analogue of basketball's "did it pass through the
        # rim": the ball at its furthest point, which is where it ended up
        # relative to the frame of the goal. Taking the most goal-central frame
        # instead would call a ball ballooned over the bar a goal, because from
        # a low camera it visually sweeps across the mouth on its way up.
        penetration = self._penetration(facts)
        if penetration >= float(self._current.get("max_penetration", 0.0)):  # type: ignore[arg-type]
            self._current["max_penetration"] = penetration
            self._current["best_entry"] = facts.goal_point
            self._current["best_entry_frame"] = facts.frame_index
            self._current["best_distance"] = facts.distance_to_frame_m
        if facts.inside_mouth:
            self._current["crossed_inside"] = True

        elapsed = facts.frame_index - int(self._current["release_frame"])  # type: ignore[arg-type]

        # 1. The ball stopped. In soccer that overwhelmingly means the net, the
        #    keeper's hands, or the ball going dead off the pitch.
        if facts.speed_mps <= self.config.stop_speed_mps and not facts.interpolated:
            self._stop_streak += 1
        else:
            self._stop_streak = 0

        if self._stop_streak >= self.config.stop_confirm_frames:
            return self._resolve(facts, reason="stopped")

        # 2. The ball reversed hard against the direction it was struck.
        #    Save, block or woodwork.
        if (
            elapsed >= 2
            and not facts.interpolated
            and facts.speed_mps >= self.config.rebound_min_speed_mps
            and self._along_shot(facts) <= self.config.rebound_cosine
        ):
            return self._resolve(facts, reason="rebound")

        # 3. We lost it, or it has been in flight implausibly long.
        if facts.interpolated and self._stop_streak == 0:
            lost = sum(1 for t in self._trace[-self.config.lost_frames :] if True)
            recent = self._trace[-self.config.lost_frames :]
            if len(recent) >= self.config.lost_frames and all(
                t.speed_mps == recent[0].speed_mps for t in recent
            ):
                return self._resolve(facts, reason="lost")
            del lost

        if elapsed >= self.config.max_shot_frames:
            return self._resolve(facts, reason="timeout")

        return None

    def force_resolve(self, facts: FrameFacts | None = None) -> Shot | None:
        """Close out an in-flight shot at end of video."""
        if self.state is not State.IN_FLIGHT:
            self.state = State.IDLE
            return None
        if facts is None and not self._trace:
            self.state = State.IDLE
            return None
        last = self._trace[-1]
        stand_in = facts or FrameFacts(
            frame_index=last.frame_index,
            position=last.position,
            velocity=(0.0, 0.0),
            speed_mps=last.speed_mps,
            goal_point=last.goal_point,
            inside_mouth=bool(self._current.get("crossed_inside")),
            distance_to_frame_m=self._current.get("best_distance"),  # type: ignore[arg-type]
            contact=None,
        )
        return self._resolve(stand_in, reason="end_of_video")

    # ------------------------------------------------------------------

    def _resolve(self, facts: FrameFacts, *, reason: str) -> Shot:
        config = self.config
        entry: Point | None = self._current.get("best_entry")  # type: ignore[assignment]
        entry_frame: int | None = self._current.get("best_entry_frame")  # type: ignore[assignment]
        distance_to_frame: float | None = self._current.get("best_distance")  # type: ignore[assignment]
        notes: list[str] = list(self._current.get("notes", []))  # type: ignore[arg-type]

        on_target = bool(
            distance_to_frame is not None and distance_to_frame >= -1e-9
        )
        zone = self.calibration.zone_of(entry) if (entry and on_target) else None

        keeper_dist_m: float | None = None
        if entry is not None and facts.keeper_positions:
            keeper_goal_points = []
            for keeper in facts.keeper_positions:
                try:
                    keeper_goal_points.append(self.calibration.to_goal_plane(keeper))
                except Exception:  # noqa: BLE001 - a keeper off-plane is not fatal
                    continue
            if keeper_goal_points:
                keeper_dist_m = min(
                    math.hypot(entry[0] - k[0], entry[1] - k[1])
                    for k in keeper_goal_points
                )

        near_woodwork = (
            on_target
            and distance_to_frame is not None
            and distance_to_frame <= config.post_tolerance_m
        )

        verdict: Verdict
        confidence = "clean"

        if reason == "rebound":
            if near_woodwork:
                verdict = "WOODWORK"
                notes.append(
                    f"rebounded within {distance_to_frame:.2f} m of the frame"
                )
            elif keeper_dist_m is not None and keeper_dist_m <= config.keeper_reach_m:
                verdict = "SAVED"
                notes.append(f"keeper within {keeper_dist_m:.2f} m of the entry point")
            elif on_target:
                verdict = "SAVED" if facts.keeper_positions else "BLOCKED"
                confidence = "inferred"
                notes.append("rebounded on target with no keeper association")
            else:
                verdict = "BLOCKED"
                notes.append("deflected before reaching the frame of the goal")
        elif reason == "stopped":
            if on_target:
                peak = max(self._speed_history) if self._speed_history else 0.0
                verdict = "GOAL"
                if facts.speed_mps > config.net_contact_speed_mps:
                    confidence = "inferred"
                    notes.append("stopped inside the mouth without a clear net decay")
                else:
                    notes.append(
                        f"speed decayed from {peak:.1f} to {facts.speed_mps:.1f} m/s inside the mouth"
                    )
                if keeper_dist_m is not None and keeper_dist_m <= config.keeper_reach_m:
                    confidence = "keeper_contact"
                    notes.append(
                        f"keeper was {keeper_dist_m:.2f} m away; could be a save on the line"
                    )
            else:
                verdict = "OFF_TARGET"
                if entry is not None:
                    notes.append(self.calibration.miss_description(entry))
        else:  # lost, timeout, end_of_video
            confidence = "occluded" if reason == "lost" else "inferred"
            if on_target and bool(self._current.get("crossed_inside")):
                verdict = "GOAL"
                notes.append(f"ball {reason} while inside the frame of the goal")
            elif on_target:
                verdict = "UNRESOLVED"
                notes.append(f"ball {reason} on target but never crossed the mouth")
            else:  # noqa: PLR5501
                verdict = "OFF_TARGET"
                notes.append(f"ball {reason} outside the frame of the goal")

        peak_speed = max(self._speed_history) if self._speed_history else 0.0
        shot = Shot(
            shot_id=int(self._current["shot_id"]),  # type: ignore[arg-type]
            verdict=verdict,
            confidence=confidence,
            start_frame=int(self._current["start_frame"]),  # type: ignore[arg-type]
            release_frame=int(self._current["release_frame"]),  # type: ignore[arg-type]
            end_frame=facts.frame_index,
            release_position=self._current["release_position"],  # type: ignore[arg-type]
            release_speed_mps=float(self._current["release_speed_mps"]),  # type: ignore[arg-type]
            peak_speed_mps=peak_speed,
            contact_keypoint=self._current.get("contact_keypoint"),  # type: ignore[arg-type]
            entry_goal_point=entry,
            entry_frame=entry_frame,
            zone=zone,
            on_target=on_target,
            distance_to_frame_m=distance_to_frame,
            trace=list(self._trace),
            notes=notes,
        )

        self.state = State.IDLE
        self._last_contact = None
        self._pre_contact_speed = 0.0
        self._stop_streak = 0
        self._trace = []
        self._speed_history = []
        self._current = {}
        return shot
