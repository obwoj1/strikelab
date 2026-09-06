"""StrikeLab Studio: the web front end.

One process serves both the API and the UI, the same shape the basketball
project uses. Uploads stream to disk, analysis runs on a background thread, and
progress reaches the browser over Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import json
import queue
import secrets
import shutil
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..constants import GOAL_HEIGHT_M, GOAL_WIDTH_M, ZONE_NAMES
from ..geometry import CalibrationError, GoalCalibration
from ..models import DetectorError
from ..physics import GROUND_PRESETS
from .jobs import JobManager
from .pipeline import extract_frame, probe_video, run_analysis
from .store import Store

DATA_DIR = Path.cwd() / "data" / "studio"
STATIC_DIR = Path(__file__).parent / "static"

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ts"}
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB

store = Store(DATA_DIR / "studio.db")
jobs = JobManager()

app = FastAPI(title="StrikeLab Studio", docs_url="/api/docs", openapi_url="/api/openapi.json")


def session_dir(session_id: str) -> Path:
    return DATA_DIR / "sessions" / session_id


def _require_session(session_id: str) -> dict[str, Any]:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="That session does not exist.")
    return session


# ----------------------------------------------------------------------
# meta


@app.get("/api/health")
def health() -> dict[str, Any]:
    from .. import __version__

    return {
        "ok": True,
        "version": __version__,
        "goal_width_m": GOAL_WIDTH_M,
        "goal_height_m": GOAL_HEIGHT_M,
        "zones": list(ZONE_NAMES),
        "ground_presets": sorted(GROUND_PRESETS),
    }


@app.get("/api/backends")
def backends() -> dict[str, Any]:
    from ..models.registry import BACKENDS

    available = []
    for choice in BACKENDS:
        if choice.name == "scripted":
            continue
        usable, reason = _backend_available(choice.name)
        available.append(
            {
                "name": choice.name,
                "description": choice.description,
                "needs_api_key": choice.needs_api_key,
                "available": usable,
                "reason": reason,
            }
        )
    return {"backends": available}


def _backend_available(name: str) -> tuple[bool, str | None]:
    if name == "ultralytics":
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return False, "Not installed. pip install 'strikelab[yolo]'"
        return True, None
    if name == "roboflow":
        import os

        if not os.environ.get("ROBOFLOW_API_KEY"):
            return False, "ROBOFLOW_API_KEY is not set"
        try:
            import inference  # noqa: F401
        except ImportError:
            return False, "Not installed. pip install 'strikelab[roboflow]'"
        return True, None
    return False, "Unknown backend"


# ----------------------------------------------------------------------
# sessions


@app.post("/api/sessions", status_code=201)
def create_session(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip() or "Untitled session"
    session_id = uuid.uuid4().hex[:12]
    session = store.create_session(session_id, name[:120])
    session_dir(session_id).mkdir(parents=True, exist_ok=True)
    return session


@app.get("/api/sessions")
def list_sessions(player_id: str | None = Query(default=None)) -> dict[str, Any]:
    return {"sessions": store.list_sessions(player_id=player_id)}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = _require_session(session_id)
    session["shots"] = store.list_shots(session_id)
    job = jobs.get(session_id)
    session["job"] = job.snapshot() if job else None
    return session


@app.patch("/api/sessions/{session_id}")
def update_session(session_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    _require_session(session_id)
    fields: dict[str, Any] = {}
    if "name" in payload:
        name = str(payload["name"] or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name cannot be empty.")
        fields["name"] = name[:120]
    if "player_id" in payload:
        player_id = payload["player_id"] or None
        if player_id and store.get_player(player_id) is None:
            raise HTTPException(status_code=400, detail="That player does not exist.")
        fields["player_id"] = player_id
    if fields:
        store.update_session(session_id, **fields)
    return _require_session(session_id)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    _require_session(session_id)
    job = jobs.get(session_id)
    if job and job.status in {"queued", "running"}:
        job.cancel()
    store.delete_session(session_id)
    shutil.rmtree(session_dir(session_id), ignore_errors=True)
    return {"deleted": True}


# ----------------------------------------------------------------------
# upload


@app.post("/api/sessions/{session_id}/video")
async def upload_video(
    session_id: str,
    request: Request,
    name: str = Query(..., min_length=1, max_length=300),
) -> dict[str, Any]:
    _require_session(session_id)

    suffix = Path(name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"{suffix or 'that file type'} is not a supported video format.",
        )

    directory = session_dir(session_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"source{suffix}"

    written = 0
    try:
        with target.open("wb") as handle:
            async for chunk in request.stream():
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="That video is too large.")
                handle.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception as error:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {error}") from error

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The upload was empty.")

    try:
        info = await asyncio.to_thread(probe_video, target)
    except Exception as error:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"That file could not be read as a video: {error}",
        ) from error

    store.update_session(
        session_id,
        video_path=str(target),
        video_name=name,
        status="uploaded",
        error=None,
        **info,
    )
    return _require_session(session_id)


@app.get("/api/sessions/{session_id}/frame")
async def get_frame(session_id: str, index: int = Query(default=0, ge=0)) -> FileResponse:
    session = _require_session(session_id)
    if not session.get("video_path"):
        raise HTTPException(status_code=409, detail="No video has been uploaded yet.")

    out_path = session_dir(session_id) / f"frame_{index:06d}.jpg"
    if not out_path.exists():
        try:
            await asyncio.to_thread(
                extract_frame, Path(session["video_path"]), index, out_path
            )
        except Exception as error:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(error)) from error
    return FileResponse(out_path, media_type="image/jpeg")


# ----------------------------------------------------------------------
# calibration


@app.post("/api/sessions/{session_id}/calibration")
def set_calibration(session_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    _require_session(session_id)

    corners = payload.get("goal_corners")
    if not isinstance(corners, list) or len(corners) != 4:
        raise HTTPException(status_code=400, detail="Provide exactly four goal corners.")
    try:
        points = [(float(p[0]), float(p[1])) for p in corners]
    except (TypeError, ValueError, IndexError) as error:
        raise HTTPException(status_code=400, detail="Corners must be [x, y] pairs.") from error

    try:
        calibration = GoalCalibration.from_corners(points)
    except CalibrationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    ground_points = payload.get("ground_points")
    ground_preset = payload.get("ground_preset") or "six-yard-box"
    if ground_points:
        if not isinstance(ground_points, list) or len(ground_points) != 4:
            raise HTTPException(status_code=400, detail="Provide exactly four ground points.")
        if ground_preset not in GROUND_PRESETS:
            raise HTTPException(status_code=400, detail=f"Unknown ground preset {ground_preset!r}.")
        try:
            ground_points = [[float(p[0]), float(p[1])] for p in ground_points]
        except (TypeError, ValueError, IndexError) as error:
            raise HTTPException(
                status_code=400, detail="Ground points must be [x, y] pairs."
            ) from error
    else:
        ground_points = None

    store.update_session(
        session_id,
        goal_corners=[list(p) for p in calibration.corners_tl_tr_br_bl],
        ground_points=ground_points,
        ground_preset=ground_preset if ground_points else None,
        status="calibrated",
    )
    session = _require_session(session_id)
    session["calibration_preview"] = {
        "corners": [list(p) for p in calibration.corners_tl_tr_br_bl],
        "zone_segments": [
            [list(a), list(b)] for a, b in calibration.zone_grid_segments()
        ],
        "pixels_per_metre": calibration.pixels_per_metre(),
    }
    return session


# ----------------------------------------------------------------------
# analysis


@app.post("/api/sessions/{session_id}/analyze", status_code=202)
def start_analysis(session_id: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    session = _require_session(session_id)

    if not session.get("video_path"):
        raise HTTPException(status_code=409, detail="Upload a video first.")
    if not session.get("goal_corners"):
        raise HTTPException(status_code=409, detail="Set the goal corners first.")
    if jobs.is_running(session_id):
        raise HTTPException(status_code=409, detail="That session is already being analysed.")

    backend = str(payload.get("backend") or "ultralytics")
    usable, reason = _backend_available(backend)
    if not usable:
        raise HTTPException(status_code=400, detail=f"Backend {backend!r} is unavailable: {reason}")

    confidence = float(payload.get("confidence") or 0.25)
    device = payload.get("device") or None
    overrides = payload.get("thresholds") or {}
    write_video = bool(payload.get("write_video", True))

    directory = session_dir(session_id)
    store.update_session(session_id, status="analysing", error=None, backend=backend)

    def work(job: Any) -> None:
        result = run_analysis(
            job,
            video_path=Path(session["video_path"]),
            output_dir=directory,
            goal_corners=session["goal_corners"],
            ground_points=session.get("ground_points"),
            ground_preset=session.get("ground_preset") or "six-yard-box",
            backend=backend,
            confidence=confidence,
            device=device,
            config_overrides=overrides,
            write_video=write_video,
        )
        store.replace_shots(session_id, result["shots"])
        store.update_session(
            session_id,
            status="cancelled" if result["cancelled"] else "done",
            summary=result["summary"],
            annotated_path=result["annotated_path"],
        )

    def on_finish(job: Any) -> None:
        if job.status == "failed":
            store.update_session(session_id, status="failed", error=job.error)

    job = jobs.start(session_id, work, on_finish=on_finish)
    return job.snapshot()


@app.post("/api/sessions/{session_id}/cancel")
def cancel_analysis(session_id: str) -> dict[str, Any]:
    _require_session(session_id)
    job = jobs.get(session_id)
    if job is None or job.status not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Nothing is running for that session.")
    job.cancel()
    return job.snapshot()


@app.get("/api/sessions/{session_id}/events")
async def analysis_events(session_id: str, request: Request) -> StreamingResponse:
    _require_session(session_id)

    async def stream() -> AsyncIterator[bytes]:
        job = jobs.get(session_id)
        if job is None:
            payload = json.dumps({"status": "idle"})
            yield f"event: progress\ndata: {payload}\n\n".encode()
            return

        listener = job.subscribe()
        try:
            yield b"retry: 3000\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.to_thread(listener.get, True, 15.0)
                except queue.Empty:
                    yield b": keep-alive\n\n"
                    continue
                if event.type == "close":
                    return
                payload = json.dumps(event.data)
                yield f"event: {event.type}\ndata: {payload}\n\n".encode()
        finally:
            job.unsubscribe(listener)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ----------------------------------------------------------------------
# media


@app.get("/api/sessions/{session_id}/annotated")
def annotated_video(session_id: str) -> FileResponse:
    session = _require_session(session_id)
    path = session.get("annotated_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="No annotated video for that session yet.")
    return FileResponse(path, media_type="video/mp4", filename="annotated.mp4")


@app.get("/api/sessions/{session_id}/export")
def export_session(session_id: str, format: str = Query(default="json")) -> Any:
    session = _require_session(session_id)
    shots = store.list_shots(session_id)

    if format == "json":
        return JSONResponse(
            {
                "session": {k: v for k, v in session.items() if k != "video_path"},
                "shots": shots,
            }
        )
    if format == "csv":
        lines = [
            "shot_id,verdict,confidence,on_target,zone,speed_kmh,quality,goal_x_m,goal_y_m,release_frame"
        ]
        for shot in shots:
            entry = shot.get("entry") or {}
            point = entry.get("goal_plane_m") or [None, None]
            quality = (shot.get("quality") or {}).get("score")
            lines.append(
                ",".join(
                    str(value if value is not None else "")
                    for value in (
                        shot.get("shot_id"),
                        shot.get("verdict"),
                        shot.get("confidence"),
                        shot.get("on_target"),
                        shot.get("zone"),
                        shot.get("peak_speed_kmh"),
                        quality,
                        point[0] if point else "",
                        point[1] if point else "",
                        (shot.get("release") or {}).get("frame"),
                    )
                )
            )
        return StreamingResponse(
            iter(["\r\n".join(lines)]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="shots.csv"'},
        )
    raise HTTPException(status_code=400, detail="format must be json or csv")


# ----------------------------------------------------------------------
# demo


@app.post("/api/demo", status_code=201)
def create_demo_session() -> dict[str, Any]:
    """Build a fully-populated session from the scripted synthetic scene.

    Runs the real engine, so the results view shows genuine output rather than
    canned JSON. It needs no model weights, which is what makes it a sane first
    thing to see on a machine with nothing set up.
    """
    from ..config import Config
    from ..engine import ShotEngine
    from ..render import Annotator
    from ..render.paint import paint_frame
    from ..sources.synthetic import SyntheticScene
    from ..sources.video import VideoWriter
    from ..stats import summarise

    session_id = uuid.uuid4().hex[:12]
    directory = session_dir(session_id)
    directory.mkdir(parents=True, exist_ok=True)
    store.create_session(session_id, "Demo session (synthetic)")

    scene = SyntheticScene()
    engine = ShotEngine(scene.calibration, config=Config(fps=30.0))
    annotator = Annotator(scene.calibration, fps=30.0)
    annotated = directory / "annotated.mp4"
    writer = VideoWriter(annotated, scene.width, scene.height, 30.0)

    frames = 0
    try:
        for scene_frame in scene.frames():
            result = engine.process(scene_frame.detections)
            canvas = paint_frame(scene_frame, scene.calibration, scene.width, scene.height)
            writer.write(annotator.draw(canvas, result))
            frames += 1
    finally:
        writer.close()
    engine.finish()

    shots = [shot.to_json() for shot in engine.shots]
    store.replace_shots(session_id, shots)
    store.update_session(
        session_id,
        status="done",
        backend="scripted",
        width=scene.width,
        height=scene.height,
        fps=30.0,
        frame_count=frames,
        video_name="synthetic.mp4",
        annotated_path=str(annotated),
        goal_corners=[list(p) for p in scene.calibration.corners_tl_tr_br_bl],
        summary=summarise(engine.shots).to_json(),
    )
    return _require_session(session_id)


# ----------------------------------------------------------------------
# players


@app.get("/api/players")
def list_players() -> dict[str, Any]:
    return {"players": store.list_players()}


@app.post("/api/players", status_code=201)
def create_player(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")
    return store.create_player(uuid.uuid4().hex[:12], name[:80])


@app.delete("/api/players/{player_id}")
def delete_player(player_id: str) -> dict[str, Any]:
    if store.get_player(player_id) is None:
        raise HTTPException(status_code=404, detail="That player does not exist.")
    store.delete_player(player_id)
    return {"deleted": True}


@app.get("/api/players/{player_id}/shots")
def player_shots(player_id: str) -> dict[str, Any]:
    player = store.get_player(player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="That player does not exist.")
    return {"player": player, "shots": store.shots_for_player(player_id)}


# ----------------------------------------------------------------------
# sharing


@app.post("/api/sessions/{session_id}/share")
def create_share(session_id: str) -> dict[str, Any]:
    session = _require_session(session_id)
    token = session.get("share_token") or secrets.token_urlsafe(16)
    store.update_session(session_id, share_token=token)
    return {"share_token": token, "url": f"/s/{token}"}


@app.delete("/api/sessions/{session_id}/share")
def revoke_share(session_id: str) -> dict[str, Any]:
    _require_session(session_id)
    store.update_session(session_id, share_token=None)
    return {"revoked": True}


@app.get("/api/shared/{token}")
def shared_session(token: str) -> dict[str, Any]:
    session = store.get_session_by_token(token)
    if session is None:
        raise HTTPException(status_code=404, detail="That share link is not valid.")
    # Never leak local filesystem paths through a public link.
    for field in ("video_path", "annotated_path"):
        session.pop(field, None)
    session["shots"] = store.list_shots(session["id"])
    return session


# ----------------------------------------------------------------------
# static UI


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/s/{token}", response_class=HTMLResponse)
def shared_page(token: str) -> HTMLResponse:  # noqa: ARG001 - the UI reads the token
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def serve(host: str = "127.0.0.1", port: int = 7878, reload: bool = False) -> None:
    import uvicorn

    uvicorn.run(
        "strikelab.web.app:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
