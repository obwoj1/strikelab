"""Model identifiers, in one place so retraining is a one-line change.

The basketball project keeps a registry.py for exactly this reason: the engine
should never care which checkpoint produced a box.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Local Ultralytics checkpoints -----------------------------------------
# These download themselves on first use. COCO already contains both classes
# this pipeline needs to get started - "sports ball" (32) and "person" (0) -
# which is why StrikeLab has a usable zero-shot path with no training at all.
YOLO_DETECT_DEFAULT = "yolo11n.pt"
YOLO_POSE_DEFAULT = "yolo11n-pose.pt"

COCO_BALL_CLASS = "sports ball"
COCO_PERSON_CLASS = "person"

# --- Roboflow-hosted models -------------------------------------------------
# Public Universe projects. Swap these for your own trained versions after
# fine-tuning on your footage; nothing else in the codebase needs to change.
ROBOFLOW_PLAYERS_MODEL = "football-players-detection-3zvbc/12"
ROBOFLOW_PITCH_KEYPOINTS_MODEL = "football-field-detection-f07vi/15"

# Class names emitted by the Roboflow players model.
ROBOFLOW_CLASS_MAP = {
    "ball": "ball",
    "player": "player",
    "goalkeeper": "goalkeeper",
    "referee": "referee",
}


@dataclass(frozen=True)
class BackendChoice:
    name: str
    description: str
    needs_network: bool
    needs_api_key: bool


BACKENDS: tuple[BackendChoice, ...] = (
    BackendChoice(
        "scripted",
        "Replays known detections. Used by `strikelab demo` and the tests.",
        needs_network=False,
        needs_api_key=False,
    ),
    BackendChoice(
        "ultralytics",
        "Local YOLO. Zero-shot on COCO for ball and people; fine-tune for goalkeepers.",
        needs_network=True,  # only to download weights the first time
        needs_api_key=False,
    ),
    BackendChoice(
        "roboflow",
        "Roboflow-hosted detection, including a soccer-specific goalkeeper class.",
        needs_network=True,
        needs_api_key=True,
    ),
)
