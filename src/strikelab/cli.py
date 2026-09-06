"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import Config
from .geometry import CalibrationError, GoalCalibration
from .physics import GROUND_PRESETS, GroundCalibration
from .types import Point


def _parse_points(raw: str, *, expected: int = 4) -> list[Point]:
    """Parse "x1,y1 x2,y2 x3,y3 x4,y4" into points."""
    chunks = raw.replace(";", " ").split()
    points: list[Point] = []
    for chunk in chunks:
        parts = chunk.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"expected 'x,y' pairs separated by spaces, got {chunk!r}"
            )
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{chunk!r} is not a numeric x,y pair") from error
    if len(points) != expected:
        raise argparse.ArgumentTypeError(
            f"expected {expected} points, got {len(points)}"
        )
    return points


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strikelab",
        description="Evaluate soccer shots from video: verdict, speed, placement and quality.",
    )
    parser.add_argument("--version", action="version", version=f"strikelab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- demo ---------------------------------------------------------
    demo = sub.add_parser(
        "demo",
        help="Run the whole pipeline on a synthetic scene. No weights, no key, no network.",
    )
    demo.add_argument("--out", default="out/demo", help="output directory")
    demo.add_argument("--fps", type=float, default=30.0)
    demo.add_argument("--no-video", action="store_true", help="skip writing the MP4")

    # --- analyze ------------------------------------------------------
    analyze = sub.add_parser("analyze", help="Analyse a real video file.")
    analyze.add_argument("video", help="path to the input video")
    analyze.add_argument(
        "--goal",
        required=True,
        help=(
            "the four corners of the goal mouth in pixels, as "
            "'x1,y1 x2,y2 x3,y3 x4,y4'. Order does not matter. "
            "Use `strikelab calibrate` to find them."
        ),
    )
    analyze.add_argument(
        "--backend",
        default="ultralytics",
        choices=["ultralytics", "roboflow"],
        help="detector to use",
    )
    analyze.add_argument("--weights", default=None, help="override detection weights")
    analyze.add_argument("--pose-weights", default=None, help="override pose weights")
    analyze.add_argument("--model-id", default=None, help="Roboflow model id")
    analyze.add_argument("--device", default=None, help="torch device, e.g. mps or cuda:0")
    analyze.add_argument("--confidence", type=float, default=0.25)
    analyze.add_argument(
        "--ground",
        default=None,
        help="four ground points in pixels for distance/angle, as 'x1,y1 ... x4,y4'",
    )
    analyze.add_argument(
        "--ground-preset",
        default="six-yard-box",
        choices=sorted(GROUND_PRESETS),
        help="which pitch markings --ground refers to",
    )
    analyze.add_argument("--out", default="out/session", help="output directory")
    analyze.add_argument("--no-video", action="store_true")
    analyze.add_argument("--max-frames", type=int, default=None)

    # --- calibrate ----------------------------------------------------
    calibrate = sub.add_parser(
        "calibrate",
        help="Save a frame so you can read off the goal corners, or click them interactively.",
    )
    calibrate.add_argument("video")
    calibrate.add_argument("--frame", type=int, default=0, help="which frame to grab")
    calibrate.add_argument("--out", default="out/calibrate", help="output directory")
    calibrate.add_argument(
        "--interactive",
        action="store_true",
        help="open a window and click the four corners (needs a GUI-capable OpenCV)",
    )

    # --- backends -----------------------------------------------------
    sub.add_parser("backends", help="List detection backends and what each needs.")

    # --- serve --------------------------------------------------------
    serve = sub.add_parser("serve", help="Run StrikeLab Studio, the web interface.")
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Interface to bind. Use 0.0.0.0 to reach it from a phone on the same "
            "network. There is no authentication, so only do that on a network you trust."
        ),
    )
    serve.add_argument("--port", type=int, default=7878)
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes")

    return parser


# ----------------------------------------------------------------------
# commands


def _cmd_backends() -> int:
    from .models.registry import BACKENDS

    print(f"{'BACKEND':<14} {'NETWORK':<9} {'API KEY':<9} DESCRIPTION")
    print("-" * 92)
    for choice in BACKENDS:
        print(
            f"{choice.name:<14} {'yes' if choice.needs_network else 'no':<9} "
            f"{'yes' if choice.needs_api_key else 'no':<9} {choice.description}"
        )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .engine import ShotEngine
    from .sources.synthetic import SyntheticScene
    from .stats import format_table, summarise, write_jsonl, write_summary

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scene = SyntheticScene()
    engine = ShotEngine(scene.calibration, config=Config(fps=args.fps))

    writer = None
    annotator = None
    if not args.no_video:
        from .render import Annotator
        from .render.paint import paint_frame
        from .sources.video import VideoWriter

        annotator = Annotator(scene.calibration, fps=args.fps)
        writer = VideoWriter(out / "demo.mp4", scene.width, scene.height, args.fps)

    frames = 0
    try:
        for scene_frame in scene.frames():
            result = engine.process(scene_frame.detections)
            frames += 1
            if writer is not None and annotator is not None:
                canvas = paint_frame(scene_frame, scene.calibration, scene.width, scene.height)
                writer.write(annotator.draw(canvas, result))
    finally:
        if writer is not None:
            writer.close()

    engine.finish()

    summary = summarise(engine.shots)
    write_jsonl(out / "shots.jsonl", engine.shots)
    write_summary(
        out / "summary.json",
        summary,
        {
            "source": "synthetic",
            "frames": frames,
            "calibration": scene.calibration.to_json(),
        },
    )

    print(format_table(engine.shots))
    print()
    print(json.dumps(summary.to_json(), indent=2))
    print()
    print(f"Wrote {out}/shots.jsonl and {out}/summary.json")
    if writer is not None:
        print(f"Wrote {out}/demo.mp4 ({frames} frames)")

    expected = scene.expected_verdicts
    actual = [shot.verdict for shot in engine.shots]
    if expected == actual:
        print(f"\nGround truth matched on all {len(expected)} scripted shots.")
    else:
        print(f"\nGround truth MISMATCH\n  expected {expected}\n  got      {actual}")
        return 1
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .engine import ShotEngine
    from .models import DetectorError, build_detector
    from .sources.video import VideoError, VideoFileSource, VideoWriter
    from .stats import format_table, summarise, write_jsonl, write_summary

    try:
        corners = _parse_points(args.goal)
        calibration = GoalCalibration.from_corners(corners)
    except (argparse.ArgumentTypeError, CalibrationError) as error:
        print(f"error: bad --goal: {error}", file=sys.stderr)
        return 2

    ground = None
    if args.ground:
        try:
            ground = GroundCalibration.from_preset(
                args.ground_preset, _parse_points(args.ground)
            )
        except (argparse.ArgumentTypeError, CalibrationError) as error:
            print(f"error: bad --ground: {error}", file=sys.stderr)
            return 2

    try:
        source = VideoFileSource(args.video)
    except VideoError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    kwargs: dict[str, object] = {"confidence": args.confidence}
    if args.backend == "ultralytics":
        if args.weights:
            kwargs["detect_weights"] = args.weights
        if args.pose_weights:
            kwargs["pose_weights"] = args.pose_weights
        if args.device:
            kwargs["device"] = args.device
    else:
        if args.model_id:
            kwargs["model_id"] = args.model_id

    try:
        detector = build_detector(args.backend, **kwargs)
    except DetectorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = Config(fps=source.info.fps)
    engine = ShotEngine(calibration, config=config, ground=ground)

    annotator = None
    writer = None
    if not args.no_video:
        from .render import Annotator

        annotator = Annotator(calibration, fps=source.info.fps)
        writer = VideoWriter(
            out / "annotated.mp4",
            source.info.width,
            source.info.height,
            source.info.fps,
        )

    processed = 0
    try:
        for index, frame in source.frames():
            if args.max_frames is not None and index >= args.max_frames:
                break
            try:
                detections = detector.detect(frame, index)
            except DetectorError as error:
                print(f"error: {error}", file=sys.stderr)
                return 1
            result = engine.process(detections)
            processed += 1
            if writer is not None and annotator is not None:
                writer.write(annotator.draw(frame, result))
            if processed % 100 == 0:
                print(f"  {processed} frames, {len(engine.shots)} shots", file=sys.stderr)
    finally:
        if writer is not None:
            writer.close()
        source.close()

    engine.finish()

    summary = summarise(engine.shots)
    write_jsonl(out / "shots.jsonl", engine.shots)
    write_summary(
        out / "summary.json",
        summary,
        {
            "source": str(Path(args.video).resolve()),
            "frames": processed,
            "backend": args.backend,
            "fps": source.info.fps,
            "calibration": calibration.to_json(),
            "ground_calibration": ground.to_json() if ground else None,
        },
    )

    print(format_table(engine.shots))
    print()
    print(json.dumps(summary.to_json(), indent=2))
    print(f"\nWrote {out}/shots.jsonl and {out}/summary.json")
    if writer is not None:
        print(f"Wrote {out}/annotated.mp4")
    if ground is None:
        print(
            "\nNote: no --ground supplied, so distance and angle are unmeasured and "
            "the quality score fell back to a 16 m prior."
        )
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .sources.video import VideoError, VideoFileSource

    try:
        import cv2
    except ImportError:
        print("error: OpenCV is required. pip install 'strikelab[video]'", file=sys.stderr)
        return 2

    try:
        source = VideoFileSource(args.video)
    except VideoError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    target = None
    for index, frame in source.frames():
        if index == args.frame:
            target = frame
            break
    source.close()

    if target is None:
        print(f"error: video has no frame {args.frame}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_path = out / f"frame_{args.frame:06d}.png"
    cv2.imwrite(str(image_path), target)

    if not args.interactive:
        print(f"Wrote {image_path}")
        print(
            "\nOpen it, read off the four corners of the goal mouth in pixels, then run:\n"
            f"  strikelab analyze {args.video} --goal 'x1,y1 x2,y2 x3,y3 x4,y4'\n"
            "\nOrder does not matter. Click the two ends of the crossbar and the two\n"
            "points where the posts meet the ground."
        )
        return 0

    clicks: list[Point] = []

    def on_click(event: int, x: int, y: int, flags: int, param: object) -> None:  # noqa: ARG001
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((float(x), float(y)))

    window = "strikelab calibrate - click the 4 goal corners, then press q"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    while True:
        preview = target.copy()
        for point in clicks:
            cv2.circle(preview, (int(point[0]), int(point[1])), 6, (80, 220, 100), -1)
        if len(clicks) == 4:
            import numpy as np

            cv2.polylines(
                preview, [np.array(clicks, dtype=np.int32)], True, (80, 220, 100), 2
            )
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or len(clicks) == 4 and key == 13:
            break
    cv2.destroyAllWindows()

    if len(clicks) != 4:
        print("error: needed 4 corners", file=sys.stderr)
        return 2

    try:
        calibration = GoalCalibration.from_corners(clicks)
    except CalibrationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    goal_arg = " ".join(f"{x:.0f},{y:.0f}" for x, y in clicks)
    (out / "goal.json").write_text(json.dumps(calibration.to_json(), indent=2) + "\n")
    print(f"Wrote {out}/goal.json")
    print(f"\nstrikelab analyze {args.video} --goal '{goal_arg}'")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from .web.app import serve
    except ImportError as error:
        print(
            f"error: the web interface needs the 'web' extra: pip install 'strikelab[web]'\n  ({error})",
            file=sys.stderr,
        )
        return 2

    if args.host not in {"127.0.0.1", "localhost"}:
        print(
            f"StrikeLab Studio is binding to {args.host}. There is no authentication, "
            "so anyone on this network can reach it. Stop the server when you are done.\n",
            file=sys.stderr,
        )
        for address in _local_addresses():
            print(f"  On your phone:  http://{address}:{args.port}", file=sys.stderr)
        print("", file=sys.stderr)

    serve(host=args.host, port=args.port, reload=args.reload)
    return 0


def _local_addresses() -> list[str]:
    """Best-effort list of this machine's LAN addresses, for the phone URL."""
    import socket

    found: set[str] = set()
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        found.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127."):
                found.add(address)
    except OSError:
        pass
    return sorted(found)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "backends":
        return _cmd_backends()
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    if args.command == "serve":
        return _cmd_serve(args)

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
