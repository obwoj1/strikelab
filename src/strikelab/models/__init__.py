from .base import Detector, DetectorError
from .registry import BACKENDS, BackendChoice
from .scripted import ScriptedDetector


def build_detector(name: str, **kwargs: object) -> Detector:
    """Construct a backend by name, importing heavy deps only when needed."""
    if name == "scripted":
        frames = kwargs.get("frames")
        if frames is None:
            raise DetectorError("the scripted backend needs a `frames` argument")
        return ScriptedDetector(frames)  # type: ignore[arg-type]
    if name == "ultralytics":
        from .ultralytics_backend import UltralyticsDetector

        return UltralyticsDetector(**kwargs)  # type: ignore[arg-type]
    if name == "roboflow":
        from .roboflow_backend import RoboflowDetector

        return RoboflowDetector(**kwargs)  # type: ignore[arg-type]

    options = ", ".join(choice.name for choice in BACKENDS)
    raise DetectorError(f"unknown backend {name!r}; available: {options}")


__all__ = [
    "BACKENDS",
    "BackendChoice",
    "Detector",
    "DetectorError",
    "ScriptedDetector",
    "build_detector",
]
