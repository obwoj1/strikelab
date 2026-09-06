# Progress

Always-current status. Readable on a phone. Updated as work lands.

**Last updated:** 2026-09-05 23:35
**Now building:** Phase 2 — persistence is in, next is charts and heatmaps
**Studio:** `strikelab serve` → http://localhost:7878
**Repo:** https://github.com/obwoj1/strikelab

---

## Where things stand

| Part | State |
|---|---|
| Analysis engine | ✅ Done, 87 tests passing |
| CLI (`demo`, `analyze`, `calibrate`) | ✅ Done |
| YOLO backend | ✅ Wired and runs on Apple MPS |
| Roboflow backend | ⚠️ Written, never run (no API key) |
| **Web app (Studio)** | ✅ Phase 1 done, running |
| Validated on real footage | ❌ Not yet — the main open risk |

---

## Plan

Four phases. Each one is shippable on its own, so stopping anywhere still
leaves something that works.

### Phase 1 — Core loop
The thing that makes the CLI bearable: drag a clip in, **click the four goal
corners** instead of typing pixel coordinates, watch it process, see results.

- [x] FastAPI app skeleton + static front end
- [x] Video upload, streamed to disk
- [x] Frame extraction for calibration
- [x] Click-to-calibrate the goal corners in the browser
- [x] Background analysis job with live progress
- [x] Results view: shot table + annotated video

### Phase 2 — Persistence and insight
- [x] SQLite storage for sessions and shots
- [x] Session history
- [x] Goal-mouth placement heatmap
- [ ] Charts: speed distribution, conversion, placement over time

### Phase 3 — Product
- [ ] Players, and assigning shots to them
- [ ] Player profiles with trend over time
- [ ] Read-only share links
- [ ] Deployment path

### Phase 4 — Make it real
- [ ] Validate against public footage
- [ ] Threshold tuning UI (sliders that re-run instantly)
- [ ] Tune defaults from what the real footage shows

---

## Open risks

1. **Never run on real soccer video.** The engine logic is tested; finding the
   ball in real footage is not. Phase 4 is where this gets settled, and it may
   send us back to the detector.
2. **Camera must be static.** Calibration is fitted once per clip. A panning
   camera breaks it. Fix is a goal-corner keypoint model.
3. **No goalkeeper class zero-shot.** `SAVED` is inferred from a rebound rather
   than observed.

---

## Session log

Newest first.

### 2026-09-05 (Sat, later)
- Built StrikeLab Studio: FastAPI + vanilla JS, one process.
- Click-to-calibrate replaces typing pixel coordinates. Drag to adjust.
- Upload streams to disk, analysis runs on a background thread, progress over SSE.
- Results: stat tiles, goal-mouth map, zone counts, shot table, annotated video.
- Players, share links, CSV/JSON export, and a one-tap demo session.
- Fixed: goal map showed 1 of 5 shots (wrong axis divisor) and mobile table overflow.

### 2026-09-05 (Sat)
- Built the engine, CLI, both detector backends, 87 tests.
- Published the repo public.
- Wrote `DEEPDIVE.md` — the basketball-to-soccer walkthrough.
- Fixed a real bug: one mis-associated frame could suppress every later shot.
- Started the web app.
