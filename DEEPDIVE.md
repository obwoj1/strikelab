# From basketball to soccer, step by step

This is the working document behind StrikeLab. It walks through how Roboflow's
[AI basketball shot evaluator](https://blog.roboflow.com/how-to-build-an-ai-basketball-shot-evaluator/)
works, then rebuilds each stage for soccer — saying at every step what carries
over unchanged, what has to be replaced, and why.

The short version: **the perception half transfers almost completely, and the
judgment half barely transfers at all.** Detecting a ball and a person is the
same problem in both sports. Deciding what happened to the ball is a different
problem, because a rim is a small hole you shoot *down through* and a goal is a
large rectangle you shoot *across*.

---

## Part 1: How the basketball evaluator works

The original splits into four components. Worth keeping this shape — it is a
good architecture and StrikeLab keeps it.

| Component | Job |
|---|---|
| **Eyes** | RF-DETR detection model (ball + rim) and RF-DETR keypoint model (pose) |
| **Memory** | Stateful engine holding position, velocity and keypoint history |
| **Judgment** | Rule-based physics deciding release, trajectory and outcome |
| **Output** | `shots.jsonl` event log plus an annotated video |

### 1.1 The models

- **Detection:** RF-DETR-small, trained on the University of Arizona
  "Basketball Shooting Robot" set (~9.8k images) from Roboflow Universe,
  filtered to two classes: **ball** and **rim**. Reported mAP50 88.84, and the
  ball found in 98.1% of in-flight frames.
- **Pose:** RF-DETR Keypoint, **zero-shot** on COCO-17. No custom training.
  Tracks wrist, elbow, shoulder, hip, knee at 0.98–1.00 confidence.

The zero-shot pose result is the most reusable finding in the whole article: a
generic COCO pose model is good enough for sports biomechanics without any
sport-specific training. That holds for soccer too.

### 1.2 Calibration

One scalar. Take the **median rim bounding-box width** across the video, assert
that a regulation rim is 18 inches (0.4572 m), and you have pixels-per-metre for
the whole clip. No checkerboard, no camera intrinsics.

This works because a rim is small, roughly circular, and roughly
fronto-parallel, so one number describes the scale everywhere it matters.

### 1.3 Release detection

A shot is declared when **all** of:

1. the ball separates from the wrist by more than 0.35 m,
2. while moving **upward**,
3. sustained for **3+ consecutive frames**,
4. and the projected parabola points at the rim's bounding box.

Condition 4 is what rejects pump fakes, wind-ups and dribbles.

### 1.4 Make or miss — the clever bit

The net **occludes the ball** for about two frames on entry. So you cannot just
watch it go through. Instead, the evaluator measures the ball's speed when it
reappears below the rim and compares it to freefall:

| Reappearance speed | Reading |
|---|---|
| 10–90% of freefall | Net friction slowed it → **make** |
| ~100% of freefall | Never touched the net → **behind-rim airball** |
| ~0% | Not the shot ball → court noise |

Rim bounces pause scoring while the ball stays within one rim-width, then
resolve as "rattled in" or "rattled out". Ambiguous calls are tagged `rattled`
or `occluded` in the log rather than being silently guessed.

### 1.5 Stated limitations

- A single 2D camera cannot always separate a front-rim bounce-out from a
  rattle-in.
- Net occlusion forces physics inference instead of direct observation.
- The rim is modelled as a **bounding box**; the article proposes training a
  rim-keypoint model on ~150 images to model it as an ellipse instead.

---

## Part 2: What actually changes for soccer

Here is the whole translation in one table, then the reasoning for each row.

| Stage | Basketball | Soccer | Transfers? |
|---|---|---|---|
| Ball detection | RF-DETR, 2 classes | Same idea; COCO `sports ball` works zero-shot | ✅ Yes |
| Pose | COCO-17 zero-shot | COCO-17 zero-shot | ✅ Yes |
| Target detection | Rim bbox | Goal mouth as a **4-corner quad** | ⚠️ Rebuilt |
| Calibration | Rim width → 1 scalar | Goal mouth → **homography** | ❌ Replaced |
| Contact joint | Wrist | **Ankle** (or head) | ⚠️ Rebuilt |
| Release trigger | Separation + **upward** | Separation + **acceleration** | ❌ Replaced |
| Outcome test | Net occlusion + freefall ratio | **Plane crossing** + deceleration | ❌ Replaced |
| Outcome classes | make / miss | GOAL / SAVED / WOODWORK / BLOCKED / OFF_TARGET | ❌ Extended |
| Key metric | Entry angle, arc | **Speed, placement, distance, angle** | ❌ Replaced |

### 2.1 Calibration: one scalar is not enough

A rim is ~0.46 m across and you film it from the side. A goal is **7.32 m wide
and 2.44 m tall**, and you almost never film it perfectly square-on. The left
post and the right post are at genuinely different depths, so a single
pixels-per-metre figure is wrong across most of the frame.

**What StrikeLab does instead:** fit a homography from the image to the plane of
the goal mouth, using the four corners.

```python
calibration = GoalCalibration.from_corners([
    (430, 196), (868, 188), (902, 372), (398, 380),
])
calibration.to_goal_plane((640, 300))   # -> (3.41, 1.05) metres
calibration.zone_of((3.41, 1.05))       # -> "middle-centre"
```

Goal-plane coordinates run 0→7.32 m from the left post and 0→2.44 m from the
ground. That single change buys three things the scalar cannot:

1. **A real containment test.** "Is this inside the frame of the goal?" is exact
   at any camera angle, instead of approximate.
2. **Placement in metres**, hence the 3×3 coaching grid — the corners are the
   "postage stamps" every finishing drill is scored on. Basketball has no
   equivalent metric: a made shot is a made shot.
3. **Perspective-correct distance to the posts and the bar**, which is what
   separates a goal from woodwork from a miss by a yard.

The `--goal` corners also work as an unexpectedly good general-purpose ruler.
The goal is the one object in a football frame whose real dimensions are fixed
by law and which is always in shot when someone is shooting at it.

### 2.2 Release: "upward" is the wrong trigger

The basketball trigger fails on soccer twice over:

- **Wrist → ankle.** Ankles are noisier and far more often occluded (by the
  other leg, by a defender). StrikeLab takes the most confident of *several*
  candidate joints — both ankles, both knees, plus the head keypoints so
  headers register — rather than relying on one clean joint.
- **Upward motion is invalid.** A soccer ball starts *on the ground* and a
  driven shot can stay under a metre for its whole flight. Requiring upward
  motion would reject most real shots.

**What replaces it:** sudden **acceleration** away from a foot, toward the goal.

```
contact within 2.6 ball diameters of a foot/head keypoint   →  ARMED
    speed ≥ 7 m/s
    AND speed ≥ 1.8 × the pre-contact baseline
    AND velocity has a goalward component
    sustained 3 consecutive frames                          →  IN_FLIGHT
```

The acceleration ratio is what separates a shot from a firm pass, and the
goalward component is what separates it from a clearance. The 3-frame sustain is
lifted directly from the basketball version — that part transfers exactly.

One subtlety that cost a real bug (see the commit "Stop one bad frame from
suppressing every later shot"): the pre-contact baseline **must be a median, not
a running average.** A single mis-associated detection produces one frame of
absurd velocity, and with an exponential average that one frame sets an
acceleration bar no real shot can clear — silently killing every subsequent
shot in the video. The tracker now also refuses any association implying more
than 55 m/s, and restarts the track instead.

### 2.3 Outcome: soccer does not need the freefall trick

This is where soccer is genuinely *easier*, and it is worth being clear about
why. The basketball evaluator has to infer a make from a friction coefficient
because **the net hides the ball at the exact moment of truth**. That is a hard
constraint and the 10–90%-of-freefall heuristic is a smart answer to it.

Soccer has no equivalent occlusion. The goal is an open rectangle; the ball is
visible as it crosses. What soccer has instead is a **depth ambiguity**: from a
2D camera you cannot directly see whether the ball crossed the goal *line* or
sailed above the bar, because on the way up it visually sweeps across the mouth.

**The naive approach fails.** My first implementation took the *most
goal-central* projection over the whole flight and called that the entry point.
That labels a ball ballooned over the crossbar a goal, because it passes
visually through the mouth on its way up.

**What works:** track the projection at the frame of **deepest penetration along
the shot direction**. That is where the ball actually ended up relative to the
goal — in the net, in the keeper's hands, past the post, or above the bar.

```
resolution triggers:
  ball stops (< 1.6 m/s for 3 frames)  →  net, keeper's hands, or dead ball
  ball reverses against shot direction →  save, block or woodwork
     (only if still moving > 3 m/s, or settling noise fires it constantly)
  ball lost / shot times out           →  low-confidence call
```

Then classify at the deepest-penetration frame:

| Trigger | Inside the mouth? | Keeper near? | Near the frame? | Verdict |
|---|---|---|---|---|
| stopped | yes | — | — | **GOAL** |
| stopped | no | — | — | **OFF_TARGET** |
| reversed | yes | — | ≤ 0.25 m | **WOODWORK** |
| reversed | yes | ≤ 1.6 m | — | **SAVED** |
| reversed | no | — | — | **BLOCKED** |
| lost / timeout | yes | — | — | GOAL, flagged `occluded` |

The deceleration check survives the translation, just repurposed. Basketball
uses it to detect **net contact through occlusion**; StrikeLab uses it to
*confirm* a goal — a ball that stops dead inside the mouth hit the net, and one
that is still travelling at 20 m/s while projecting inside the mouth has not
arrived yet. Same physics, different question.

The confidence tags carry over wholesale, and they are the most underrated part
of the original design: `clean`, `keeper_contact`, `inferred`, `occluded`. An
honest "I think this was a goal but the keeper was on the line" beats a
confident wrong answer.

### 2.4 More outcome classes, and why it matters

Basketball is binary. Soccer is not, and collapsing it to
scored/didn't-score throws away the coaching signal:

- **SAVED** vs **OFF_TARGET** is the single most useful split in finishing
  practice. Hitting the target and being denied is a different problem from
  missing.
- **WOODWORK** is process-good, outcome-bad. A striker who hits the post twice
  had a better session than the numbers say.
- **BLOCKED** is a shot-selection problem, not a finishing problem.

Distinguishing SAVED from GOAL is the one place StrikeLab genuinely needs a
soccer-specific model, because **COCO has no goalkeeper class.** Zero-shot, the
keeper is just another `person`. That is the honest limit of the no-training
path, and the main reason to fine-tune.

### 2.5 Metrics that have no basketball analogue

The article computes entry angle and arc peak. Those are the right metrics for
basketball and mostly meaningless for soccer. What soccer wants:

- **Shot speed in km/h.** The headline number for any striker, and basketball
  ignores it entirely. Derived from pixel motion divided by the goal-plane
  scale.
- **Placement zone.** Which of the 9 cells the ball crossed. Directly coachable.
- **Distance and angle.** Optional, needs a ground-plane homography — supply
  four points on a known marking (`--ground` with the six-yard box or penalty
  area preset). Without it, distance is reported as `null` and the quality score
  falls back to a declared 16 m prior rather than inventing a number.
- **Shot quality.** A transparent 0–1 score from distance, angle, placement and
  speed. **Deliberately not called xG**, because it was hand-tuned rather than
  fitted to a shot database. Every term is reported alongside the score so a
  coach can disagree with the weighting and still use the components.

### 2.6 Camera geometry differs, and it matters

Basketball footage is shot close, from the side, in a fixed gym. Soccer footage
is filmed from further away, at wider angles, in changing light. Three knock-on
effects:

- **The ball is much smaller in frame**, often 8–15 px. Detection recall is the
  binding constraint, not classification accuracy.
- **The ball is occluded more**, by legs, defenders and the keeper. Hence the
  tracker's gap-coasting: a leg crossing in front of the ball must not drop the
  track mid-shot.
- **A whole extra failure mode: the second ball.** Training grounds have spare
  balls lying around, and a stationary ball on the touchline will happily
  hijack the track. Hence the association gate rejecting any candidate more than
  0.55 goal-widths from the prediction.

---

## Part 3: The models, concretely

### Zero-shot path (no training, works today)

COCO already contains **`sports ball` (32)** and **`person` (0)**, so
`yolo11n.pt` plus `yolo11n-pose.pt` gives ball tracking, contact detection and
skeletons with nothing to train:

```bash
strikelab analyze match.mp4 --goal '430,196 868,188 902,372 398,380'
```

What you lose: no goalkeeper class, so SAVED is inferred from a rebound rather
than observed, and no goal detection, so you pass `--goal` yourself.

### Fine-tuned path (what the basketball article actually does)

Mirror the original: one small detection model, sport-specific classes.

| Need | Source |
|---|---|
| ball / player / **goalkeeper** / referee | [`football-players-detection-3zvbc`](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) |
| pitch keypoints for ground calibration | [`football-field-detection-f07vi`](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) |
| pose | COCO-17 zero-shot, exactly as the article found |

Model IDs live in `models/registry.py`, the same pattern the basketball repo
uses, so swapping in your own trained weights is a one-line change.

### The one model worth training yourself

The basketball article ends by proposing a **rim-keypoint model on ~150 images**
to replace the bounding box with an ellipse. The soccer version of that idea is
the highest-value training you can do here: a **goal-keypoint model** predicting
the four corners of the goal mouth, on a couple of hundred images.

That would remove the `--goal` flag entirely and, more importantly, make
calibration **per-frame**, which is what you need for a panning or handheld
camera. Right now StrikeLab assumes a static camera, which is the single biggest
limitation in the whole pipeline.

---

## Part 4: What I would build next

In priority order, with reasoning.

1. **Goal-corner keypoint model** (~200 images). Kills the manual `--goal` flag
   and unlocks moving cameras. Highest value per unit of work by a distance.
2. **Fine-tune for the goalkeeper class.** Turns SAVED from inferred to
   observed, and it is the difference between "the ball stopped in the mouth"
   and "the keeper caught it".
3. **Pitch-keypoint homography** for automatic distance and angle, so the
   quality score stops falling back to a prior.
4. **Fit the quality model properly.** Right now it is hand-tuned and labelled
   as such. With a few hundred logged shots it could be a real logistic
   regression, and then it would deserve to be called xG.
5. **Ball-size depth cue.** A ball is 0.22 m; its pixel diameter relative to its
   expected size at the goal plane is a weak depth signal that would help
   disambiguate over-the-bar from into-the-net without any ground calibration.

---

## Appendix: things the basketball article got right that I copied wholesale

- **Frame-in / events-out engine, decoupled from the video source.** It is why
  the test suite can drive the whole pipeline with scripted detections and no
  weights.
- **A `registry.py` for model IDs.** The engine should never care which
  checkpoint produced a box.
- **Confidence flags on every verdict.** `clean` / `keeper_contact` /
  `inferred` / `occluded`.
- **JSONL event log plus a rendered video.** One for analysis, one for the
  person who has to actually believe the numbers.
- **Zero-shot COCO pose.** Confirmed: no sport-specific pose training needed.
