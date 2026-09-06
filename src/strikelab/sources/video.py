"""Video input and output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:  # pragma: no cover
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


class VideoError(RuntimeError):
    pass


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise VideoError(
            "OpenCV is required to read or write video. pip install strikelab[video]"
        )


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


class VideoFileSource:
    """Reads a file frame by frame."""

    def __init__(self, path: str | Path) -> None:
        _require_cv2()
        self.path = Path(path)
        if not self.path.exists():
            raise VideoError(f"no such video: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise VideoError(
                f"could not open {self.path}. Is it a video file OpenCV can read?"
            )

        fps = float(self._capture.get(cv2.CAP_PROP_FPS)) or 30.0
        self.info = VideoInfo(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps > 1.0 else 30.0,
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        index = 0
        try:
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    return
                yield index, frame
                index += 1
        finally:
            self._capture.release()

    def close(self) -> None:
        self._capture.release()


class VideoWriter:
    """Writes an annotated MP4."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float) -> None:
        _require_cv2()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise VideoError(f"could not open {self.path} for writing")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
