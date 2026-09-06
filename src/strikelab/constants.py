"""Laws-of-the-Game measurements, in metres.

These are the soccer equivalent of the basketball evaluator's "a regulation rim
is 18 inches wide" calibration anchor. A rim is a small circle you have to shoot
*down* through; a goal is a large rectangle you have to shoot *across*. That one
difference drives most of the design in this package.
"""

from __future__ import annotations

# Goal mouth (IFAB Law 1). The single most useful calibration reference in the
# frame: it is large, rigid, always visible in shooting footage, and its real
# dimensions never change.
GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44

# Ball (IFAB Law 2, size 5): circumference 68-70 cm.
BALL_DIAMETER_M = 0.22

# Pitch markings used for optional ground-plane calibration. Distances are
# measured from the goal line; widths are the full span, centred on the goal.
SIX_YARD_BOX_DEPTH_M = 5.5
SIX_YARD_BOX_WIDTH_M = 18.32
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
PENALTY_SPOT_DEPTH_M = 11.0

GRAVITY_MPS2 = 9.81

# Placement grid across the goal mouth, as used by coaches: the corners are the
# "postage stamps". Column and row order is left-to-right, bottom-to-top from
# the camera's point of view.
ZONE_COLUMNS = 3
ZONE_ROWS = 3

ZONE_NAMES: tuple[str, ...] = (
    "bottom-left",
    "bottom-centre",
    "bottom-right",
    "middle-left",
    "middle-centre",
    "middle-right",
    "top-left",
    "top-centre",
    "top-right",
)

# Corner zones are the hardest for a keeper to reach, centre zones the easiest.
# Used by the xG-lite placement modifier.
ZONE_DIFFICULTY: dict[str, float] = {
    "bottom-left": 1.35,
    "bottom-centre": 0.70,
    "bottom-right": 1.35,
    "middle-left": 1.15,
    "middle-centre": 0.55,
    "middle-right": 1.15,
    "top-left": 1.50,
    "top-centre": 0.90,
    "top-right": 1.50,
}
