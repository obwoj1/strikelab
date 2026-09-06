"""Tunable thresholds for the shot engine.

Every number that decides "was that a shot?" or "did it go in?" lives here, so
tuning against your own footage never means editing engine logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class TrackerConfig:
    """Ball association and smoothing."""

    # Reject a candidate ball detection further than this from the prediction,
    # as a fraction of the goal width. Keeps stray balls on the touchline from
    # hijacking the track.
    max_association_dist_goal_widths: float = 0.55
    # How many frames the ball may go missing before the track is dropped.
    max_gap_frames: int = 12
    # Exponential smoothing on position. 1.0 disables smoothing.
    position_alpha: float = 0.65
    # Velocity is averaged over this many frames to survive one-frame jitter.
    velocity_window: int = 3
    min_confidence: float = 0.25


@dataclass(frozen=True)
class ShotConfig:
    """What counts as a shot, and when it is over."""

    # A player's foot (ankle keypoint) or head must be within this distance of
    # the ball for a contact to arm the state machine. In ball diameters.
    contact_radius_ball_diameters: float = 2.6
    # Contact must have happened within this many frames of the speed spike.
    contact_memory_frames: int = 8

    # Release requires the ball to exceed this speed. Basketball uses "ball
    # separates from the wrist while moving upward"; a soccer ball starts on
    # the ground and often stays low, so upward motion is not a valid trigger.
    # Sudden acceleration is.
    release_speed_mps: float = 7.0
    # ...and to have accelerated by at least this factor over its pre-contact
    # speed, which is what separates a shot from a firm pass or a dribble.
    release_accel_ratio: float = 1.8
    # Sustained for this many consecutive frames.
    release_confirm_frames: int = 3
    # The velocity must have a positive component toward the goal plane.
    min_goalward_cosine: float = 0.20

    # Resolution triggers.
    stop_speed_mps: float = 1.6
    stop_confirm_frames: int = 3
    # A reversal this strong against the shot direction ends the shot...
    rebound_cosine: float = -0.35
    # ...but only if the ball is still genuinely moving. A ball settling in the
    # net or rolling to a halt reverses direction at walking pace all the time;
    # a ball coming back off a keeper or a post does not.
    rebound_min_speed_mps: float = 3.0
    lost_frames: int = 15
    max_shot_frames: int = 240

    # Classification.
    # How close to a post or the crossbar still counts as woodwork, in metres.
    post_tolerance_m: float = 0.25
    # A keeper this close to the ball's goal-plane entry point gets the credit.
    keeper_reach_m: float = 1.6
    # Below this speed after crossing the mouth, we call it net contact.
    net_contact_speed_mps: float = 3.0

    # Minimum plausible shot speed for a metric to be reported at all.
    min_reportable_speed_mps: float = 4.0


@dataclass(frozen=True)
class RenderConfig:
    draw_skeleton: bool = True
    draw_trail: bool = True
    trail_frames: int = 28
    draw_goal_mesh: bool = True
    draw_zone_grid: bool = True
    draw_hud: bool = True
    font_scale: float = 0.5


@dataclass(frozen=True)
class Config:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    shot: ShotConfig = field(default_factory=ShotConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    fps: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
