"""Local YOLO backend.

The useful accident here is that COCO already contains "sports ball" and
"person", so this gives a working soccer pipeline with zero training. What COCO
does *not* give you is the goalkeeper, which matters because separating the
keeper from the outfield players is what turns "the ball stopped inside the
mouth" into "saved" rather than "goal". See `strikelab train --help` and the
README for the fine-tune path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import COCO_KEYPOINT_NAMES, Box, FrameDetections, Pose
from .base import Detector, DetectorError
from .registry import (
    COCO_BALL_CLASS,
    COCO_PERSON_CLASS,
    YOLO_DETECT_DEFAULT,
    YOLO_POSE_DEFAULT,
)


class UltralyticsDetector(Detector):
    name = "ultralytics"

    def __init__(
        self,
        *,
        detect_weights: str = YOLO_DETECT_DEFAULT,
        pose_weights: str | None = YOLO_POSE_DEFAULT,
        ball_class: str = COCO_BALL_CLASS,
        person_class: str = COCO_PERSON_CLASS,
        goalkeeper_class: str = "goalkeeper",
        confidence: float = 0.25,
        device: str | None = None,
        imgsz: int = 960,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - depends on extras
            raise DetectorError(
                "The ultralytics backend needs the 'yolo' extra. "
                "Install it with: pip install 'strikelab[yolo]'"
            ) from error

        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
        self.ball_class = ball_class
        self.person_class = person_class
        self.goalkeeper_class = goalkeeper_class

        try:
            self.detector = YOLO(detect_weights)
            self.pose = YOLO(pose_weights) if pose_weights else None
        except Exception as error:  # noqa: BLE001 - surface a readable message
            raise DetectorError(
                f"Could not load YOLO weights ({detect_weights!r}"
                f"{', ' + repr(pose_weights) if pose_weights else ''}): {error}"
            ) from error

    # ------------------------------------------------------------------

    def _predict(self, model: Any, frame: np.ndarray) -> Any:
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        return model.predict(frame, **kwargs)[0]

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameDetections:
        result = self._predict(self.detector, frame)
        names = result.names

        ball: Box | None = None
        players: list[Box] = []
        keepers: list[Box] = []

        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for row in boxes:
                label = str(names[int(row.cls[0])])
                confidence = float(row.conf[0])
                x1, y1, x2, y2 = (float(v) for v in row.xyxy[0])
                box = Box(x1, y1, x2, y2, confidence, label)

                if label == self.ball_class:
                    # Keep only the most confident ball; the engine tracks one.
                    if ball is None or confidence > ball.confidence:
                        ball = box
                elif label == self.goalkeeper_class:
                    keepers.append(box)
                elif label in {self.person_class, "player"}:
                    players.append(box)

        poses: list[Pose] = []
        if self.pose is not None:
            pose_result = self._predict(self.pose, frame)
            keypoints = getattr(pose_result, "keypoints", None)
            if keypoints is not None and keypoints.data is not None:
                for person in keypoints.data:
                    array = person.cpu().numpy() if hasattr(person, "cpu") else np.asarray(person)
                    mapping: dict[str, tuple[float, float, float]] = {}
                    for index, name in enumerate(COCO_KEYPOINT_NAMES):
                        if index >= len(array):
                            break
                        x, y = float(array[index][0]), float(array[index][1])
                        confidence = float(array[index][2]) if len(array[index]) > 2 else 1.0
                        mapping[name] = (x, y, confidence)
                    if mapping:
                        poses.append(Pose(keypoints=mapping))

        return FrameDetections(
            frame_index=frame_index,
            ball=ball,
            players=players,
            goalkeepers=keepers,
            poses=poses,
        )
