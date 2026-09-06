# StrikeLab

**Point a camera at a goal, get a verdict on every shot.**

StrikeLab watches soccer footage and, for each attempt, tells you what happened
to the ball, how hard it was struck, exactly where it crossed the goal mouth,
and how good a chance it was. It writes a JSON event log and an annotated video.

```
  #  VERDICT         SPEED  PLACEMENT           Q  NOTE
-------------------------------------------------------
  1  GOAL            50 kmh  bottom-left     0.142  speed decayed from 13.9 to 0.1 m/s inside the mouth
  2  SAVED           52 kmh  bottom-centre   0.075  keeper within 0.65 m of the entry point
  3  OFF_TARGET      66 kmh  off target      0.047  1.8 m wide left
  4  GOAL            61 kmh  top-right       0.169  speed decayed from 16.9 to 0.2 m/s inside the mouth
  5  OFF_TARGET      87 kmh  off target      0.052  1.7 m over
```

It is a soccer rebuild of Roboflow's
[AI basketball shot evaluator](https://blog.roboflow.com/how-to-build-an-ai-basketball-shot-evaluator/).
The perception half of that design transfers almost unchanged; the judgment half
does not, because a rim is a small hole you shoot *down through* and a goal is a
large rectangle you shoot *across*. **[DEEPDIVE.md](DEEPDIVE.md) walks through
the original step by step and then rebuilds each stage for soccer**, which is
the more interesting half of this repo.

---

## See it work in 30 seconds

No model weights, no API key, no network:

```bash
git clone https://github.com/obwoj1/strikelab.git
cd strikelab
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,video]'
.venv/bin/python -m strikelab demo
```

That renders a scripted five-shot training-ground scene, runs the real engine
over it, and writes `out/demo/demo.mp4`, `shots.jsonl` and `summary.json`. The
scene has known ground truth, so the command tells you whether the engine got
all five right.

## What you get per shot

```json
{
  "shot_id": 1,
  "verdict": "GOAL",
  "confidence": "clean",
  "on_target": true,
  "zone": "bottom-left",
  "release": { "frame": 27, "speed_kmh": 48.3, "contact_keypoint": "left_knee" },
  "peak_speed_kmh": 50.2,
  "entry": { "frame": 42, "goal_plane_m": [0.843, 0.593], "distance_to_frame_m": 0.593 },
  "duration_s": 0.5,
  "quality": { "score": 0.1423, "distance_term": 0.1726, "placement_term": 1.35 },
  "notes": ["speed decayed from 13.9 to 0.1 m/s inside the mouth"]
}
```

**Verdicts:** `GOAL`, `SAVED`, `WOODWORK`, `BLOCKED`, `OFF_TARGET`, `UNRESOLVED`.
Every one carries a confidence tag — `clean`, `keeper_contact`, `inferred` or
`occluded` — so a guess never looks like an observation.

`goal_plane_m` is metres from the left post and metres above the ground, and
`zone` is the cell of the 3×3 coaching grid the ball crossed. That is the number
a finishing session is actually scored on.

## Running it on your own footage

Two steps. First find the four corners of the goal mouth:

```bash
.venv/bin/python -m strikelab calibrate match.mp4 --frame 0
# writes a PNG; read off the corners, or use --interactive to click them
```

Then analyse:

```bash
.venv/bin/python -m strikelab analyze match.mp4 \
  --goal '430,196 868,188 902,372 398,380' \
  --backend ultralytics --device mps
```

Optionally add distance and shot angle by calibrating the ground plane against a
marking whose real dimensions are fixed by law:

```bash
  --ground '200,500 800,500 740,380 260,380' --ground-preset six-yard-box
```

Without `--ground`, distance and angle are reported as `null` and the quality
score falls back to a **declared** 16 m prior rather than inventing a number.

## Detection backends

```bash
.venv/bin/python -m strikelab backends
```

| Backend | Needs | Notes |
|---|---|---|
| `scripted` | nothing | Replays known detections. Powers `demo` and the tests. |
| `ultralytics` | `pip install '.[yolo]'` | Local YOLO. **Zero-shot** — COCO already has `sports ball` and `person`. |
| `roboflow` | `ROBOFLOW_API_KEY` | Adds a real **goalkeeper** class, which COCO does not have. |

The zero-shot path works with no training at all, which is the nice surprise
here. Its honest limit is the goalkeeper: without that class a save is *inferred*
from a rebound rather than observed. Model IDs live in `models/registry.py` so
swapping in your own fine-tune is a one-line change.

## How it decides

Four stages, the same architecture the basketball evaluator uses:

- **Eyes** — a detector finds the ball and people; a COCO-17 pose model finds
  ankles, knees and head. Pose needs no sport-specific training.
- **Calibration** — the four goal corners fit a **homography** onto the plane of
  the goal mouth. A single pixels-per-metre scalar (which is all a basketball rim
  needs) is not enough for a 7.32 m goal seen at an angle.
- **Release** — a contact near a foot arms the machine; a **sustained
  acceleration** with a goalward component fires it. "Ball moves upward" is the
  basketball trigger and is wrong here: a soccer ball starts on the ground and a
  driven shot can stay under a metre the whole way.
- **Resolution** — the verdict is read at the frame of **deepest penetration
  along the shot direction**, not the most goal-central frame. From a low camera
  a ball ballooned over the bar visually sweeps across the mouth on the way up,
  and reading the most central frame calls that a goal.

## Tests

```bash
.venv/bin/python -m pytest        # 87 tests
```

Pure geometry, physics, tracking and state-machine logic are unit tested. The
engine is also driven end to end against the scripted scene, so a change that
alters a verdict fails a test rather than quietly changing someone's numbers.

## Known limits

Read this part before trusting a number.

- **The camera must be static.** Calibration is fitted once from `--goal` and
  reused for the whole clip. Pan, tilt or zoom and the goal plane is wrong. This
  is the single biggest limitation; the fix is a goal-corner keypoint model (see
  DEEPDIVE.md Part 4).
- **Zero-shot cannot see a goalkeeper.** COCO has no such class, so on the
  `ultralytics` backend a keeper is just another person and `SAVED` is inferred
  from a rebound. Use the `roboflow` backend or fine-tune for a real save call.
- **Depth is not observed.** A 2D camera cannot see the goal line. Whether the
  ball crossed is inferred from where it stopped, not from watching it cross.
  Marginal over-the-bar and off-the-line calls are the weak spot.
- **Speed is measured at the goal-plane scale**, so a ball much nearer the camera
  than the goal reads fast. The scale used is recorded in `summary.json`.
- **The quality score is hand-tuned, not fitted.** That is why it is not called
  xG. Every term is reported next to it so you can disagree with the weighting.
- **Tested on synthetic footage only.** See below.

## What has and has not been verified

Being straight about this, because it is a computer vision project and the
detection half is the part that usually disappoints.

**Verified:** the engine, tracker, geometry, physics and state machine, against
scripted detections with known ground truth (87 passing tests, all five demo
shots correct). The `ultralytics` backend is confirmed to load weights, run on
Apple MPS, and correctly parse boxes, classes and COCO keypoints.

**Not verified:** accuracy on real match or training-ground footage. I had no
soccer video to test against, so no claim is made about real-world detection
recall, and the thresholds in `config.py` are reasoned defaults rather than
values tuned on real shots. Expect to tune `release_speed_mps` and
`contact_radius_ball_diameters` for your camera distance. On the cartoon demo
video YOLO detects the ball at only ~0.5 confidence and finds no shots at all —
that is a property of the synthetic input, not a bug, and it is exactly why the
demo uses the `scripted` backend.

**Not implemented:** the `roboflow` backend is written against the documented
inference SDK but has never been run, as I had no API key.

## Layout

```
src/strikelab/
  constants.py    Laws of the Game measurements
  config.py       every tunable threshold, in one place
  geometry.py     goal-plane homography, containment, 3x3 zones
  physics.py      ground calibration, shot angle, quality score
  engine/         tracker, state machine, frame-in/events-out engine
  models/         detector backends + registry.py
  sources/        video IO and the scripted synthetic scene
  render/         annotated overlay
  stats/          JSONL log, session summary, terminal table
```

## Credits

The architecture, the four-component split, the confidence tagging and the
`registry.py` pattern are lifted from Roboflow's basketball shot evaluator by
[Aarnav Shah](https://github.com/aarnavshah12/shot-tracker), written up
[here](https://blog.roboflow.com/how-to-build-an-ai-basketball-shot-evaluator/).
The soccer geometry, state machine and metrics are this repo's own.

MIT licensed.
