# Progress

Always-current status. Readable on a phone. Updated as work lands.

**Last updated:** 2026-09-05 23:05
**Now building:** StrikeLab Studio — the web front end
**Repo:** https://github.com/obwoj1/strikelab

---

## Where things stand

| Part | State |
|---|---|
| Analysis engine | ✅ Done, 87 tests passing |
| CLI (`demo`, `analyze`, `calibrate`) | ✅ Done |
| YOLO backend | ✅ Wired and runs on Apple MPS |
| Roboflow backend | ⚠️ Written, never run (no API key) |
| **Web app** | 🔨 In progress |
| Validated on real footage | ❌ Not yet — the main open risk |

---

## Plan

Four phases. Each one is shippable on its own, so stopping anywhere still
leaves something that works.

### Phase 1 — Core loop
The thing that makes the CLI bearable: drag a clip in, **click the four goal
corners** instead of typing pixel coordinates, watch it process, see results.

- [ ] FastAPI app skeleton + static front end
- [ ] Video upload, streamed to disk
- [ ] Frame extraction for calibration
- [ ] Click-to-calibrate the goal corners in the browser
- [ ] Background analysis job with live progress
- [ ] Results view: shot table + annotated video

### Phase 2 — Persistence and insight
- [ ] SQLite storage for sessions and shots
- [ ] Session history
- [ ] Goal-mouth placement heatmap
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

### 2026-09-05 (Sat)
- Built the engine, CLI, both detector backends, 87 tests.
- Published the repo public.
- Wrote `DEEPDIVE.md` — the basketball-to-soccer walkthrough.
- Fixed a real bug: one mis-associated frame could suppress every later shot.
- Started the web app.
