"""Data carried between the detector, the engine and the renderer.

The engine is deliberately frame-in / events-out and knows nothing about which
model produced these. That is what lets the same engine run against YOLO,
RF-DETR, or a scripted fixture in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

Point = tuple[float, float]

# COCO-17 names, shared by every pose model we support.
ANKLE_KEYPOINTS = ("left_ankle", "right_ankle")
HEAD_KEYPOINTS = ("nose", "left_eye", "right_eye")
CONTACT_KEYPOINTS = ANKLE_KEYPOINTS + HEAD_KEYPOINTS + ("left_knee", "right_knee")

COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_SKELETON: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

Verdict = Literal["GOAL", "SAVED", "WOODWORK", "BLOCKED", "OFF_TARGET", "UNRESOLVED"]


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    label: str = ""

    @property
    def centre(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def bottom_centre(self) -> Point:
        """Where a standing person meets the ground; used for pitch distance."""
        return ((self.x1 + self.x2) / 2.0, max(self.y1, self.y2))

    def to_json(self) -> dict[str, float | str]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "confidence": round(self.confidence, 4),
            "label": self.label,
        }


@dataclass(frozen=True)
class Pose:
    """One person's keypoints, keyed by COCO name."""

    keypoints: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    box: Box | None = None

    def point(self, name: str, min_confidence: float = 0.3) -> Point | None:
        found = self.keypoints.get(name)
        if found is None:
            return None
        x, y, confidence = found
        if confidence < min_confidence:
            return None
        return (x, y)

    def any_point(
        self, names: Iterable[str], min_confidence: float = 0.3
    ) -> tuple[str, Point] | None:
        """The most confident of a group of keypoints, e.g. either ankle."""
        best: tuple[str, Point] | None = None
        best_confidence = min_confidence
        for name in names:
            found = self.keypoints.get(name)
            if found is None:
                continue
            x, y, confidence = found
            if confidence >= best_confidence:
                best_confidence = confidence
                best = (name, (x, y))
        return best


@dataclass
class FrameDetections:
    """Everything a detector found in one frame."""

    frame_index: int
    ball: Box | None = None
    players: list[Box] = field(default_factory=list)
    goalkeepers: list[Box] = field(default_factory=list)
    goal: Box | None = None
    poses: list[Pose] = field(default_factory=list)

    @property
    def all_people(self) -> list[Box]:
        return [*self.players, *self.goalkeepers]
