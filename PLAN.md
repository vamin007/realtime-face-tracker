# Plan: Object Tracker v2

## Status (2026-10-17): M0–M6 done on the `v2` branch
Measured on the reference clips, with everything on:

| | Baseline (`main`) | v2 |
|---|---|---|
| Live speed (`--realtime`, labels on) | ~10 FPS | 30.0 FPS (camera's maximum), no skipped frames |
| `hands_face.mov` hand IDs | 75 | 4 |
| `flashlight.mov` hand IDs | 25 | 4 |
| Face IDs | 7 (shown as "person") | 1 per clip |
| TARGET on flashlight episodes | — | 10 / 10, each within 0.2 s; none in `hands_face.mov` |
| Custom labels (photos from the clips) | — | face recognized in 99% of frames, phone in 75% |
| Unit tests | 0 | 71 |

Added on 2026-10-20 and merged into `v2` (details in the sections below):
- **Lag and flicker** (`v2-tracking`): boxes on fast-moving hands trail their detection by 4.0–4.4% of the box size instead of 8.8–11.4%, and switch to a thin (predicted) box 24 times a minute instead of 57 on `hands_face`. Hand IDs, 30 FPS and the label rates are unchanged.
- **Label database** (`v2-labels-db`, under M5): each label is a folder of photos in `labels/`, synced at startup. 17 new unit tests (88 on that branch).
- **UI polish** (`v2-ui`): a new overlay with tags kept off each other, corner-only predicted boxes, a TARGET LOCKED badge, a HUD panel and `h` for help. Drawing takes about 0.2 ms per frame (0.5 ms with the debug view and trails).

Still to do by the user, with the live camera (no clip covers these):
- a live run: camera capture runs on a background thread now;
- the UI polish overlay in a live window: the keys, pausing, the help panel, the pulsing TARGET corners;
- enrolling a real object from 5–10 photos;
- telling two different items of the same class apart.

Each milestone section below starts with what was built and measured. The original plan text follows it.

## Context
On live camera the tracker runs at about 10 FPS. Its IDs keep changing: the 62 s run in `logs/run_20261004_214134.csv` created 139 track IDs, 18 of them for two hands. "person" secretly means a face, and the project has no git repo and no tests. This plan fixes those problems and adds four features: higher FPS, numbering per type with face and hand as separate types, custom labels learned from photos, and bright-light face targeting. The goal is stable IDs at about 30 FPS with all of those features.

Decisions made (2026-10-14):
- Custom labels: photo folder plus a `--enroll` command.
- Labels recognize specific instances of things the detectors already find.
- Only the brightest lit face is targeted.
- ONNX Runtime with CoreML for speed.
- Numbers per type are never reused within a run.

Claude writes the code and explains each concept as the milestone reaches it.

## Review → fix map
| # | Issue / request | From | Fixed in |
|---|---|---|---|
| 1 | IDs keep changing (drift while unseen, overlap-only matching, label flips break tracks) | Claude | M3 |
| 2 | "person" means a face; faces and hands should be their own objects | Both | M1 |
| 3 | About 10 FPS; higher FPS makes tracking easier | Both | M2 |
| 4 | `Detector` does too much, circular includes, class-ID workaround, one threshold for three models | Claude | M1 |
| 5 | Robustness: out-of-bounds class names, bad CLI errors, silent fallbacks, video plays too fast | Claude | M1, M6 |
| 6 | No git, no tests, no README, models can't be re-downloaded, mixed formatting | Claude | M0 |
| 7 | Numbers per type (`hand #1`, `face #1`) | User | M3 |
| 8 | Custom labels from several photos, recognized by a neural network | User | M5 |
| 9 | Bright light on a face → red box labeled `TARGET` | User | M4 |
| 10 | General polish | User | M6 |

## Target architecture
```
Capture thread ──latest frame──▶ Main loop (per frame)
                                  ├─ Detectors in parallel ── YOLO (ORT+CoreML) │ Face YuNet (OpenCV) │ Hand palm (ORT+CoreML)
                                  ├─ Tracker: Kalman predict → Hungarian matching (2 passes) → track states → numbers per type
                                  ├─ TargetSelector: face brightness ÷ background brightness → one TARGET
                                  ├─ Recognition worker (async): crop → embedding → closest label → votes per track
                                  └─ Overlay + HUD │ Logger (CSV) │ Recorder (optional)
```

Layout: `src/{app,core,io,infer,detect,track,recog,target,ui}/`, plus `tests/`, `scripts/`, `labels/` and `clips/`. (As built, `src/` stayed flat, and `tools/` holds `replay.cpp` and `grab_frames.cpp`; see the README's Layout.)

## Milestones
Each milestone builds, runs and gets its own git commit.

### M0: Foundation (no behavior changes)
- `git init` and commit the current code as the baseline. `.gitignore` covers `build/`, `logs/`, `models/*.onnx`, `.venv/` and `third_party/`.
- CMake split: a `tracker_core` library, the `tracker` executable and a `tracker_tests` target (GoogleTest from `brew install googletest`, run through `ctest`).
- `.clang-format` with 4-space indents to match the other files. Reformat `Tracker.cpp`, the only file that differs.
- `scripts/fetch_models.sh` downloads the models (YuNet, palm, SFace, DINOv2) with checksums. `scripts/export_yolo.sh` exports YOLO using a project-local `.venv`.
- `README.md`: build, run, enroll, flags.
- `scripts/log_stats.py` reads a run's CSV and prints frames, FPS, confirmed IDs per class and mean track lifetime. `--stats` doesn't exist until M2, so the baseline needs this. It's updated in M6 for the new CSV columns.
- **Baseline:** record two reference clips with QuickTime (it saves `.mov`, which OpenCV reads):
  - `clips/hands_face.mov`: 60 s of hands, face and a cup.
  - `clips/flashlight.mov`: a phone flashlight on a face, then off.

  Run the current code on them with `--video`, then run `log_stats.py` on the logs and save the numbers to compare against later.

### M1: Detection rebuild and separate face/hand objects
- `Detection.h` in its own header: box, class, confidence, which detector produced it, and keypoints (face landmarks are needed for recognition in M5). This breaks the circular include between `Detector.h` and `FaceDetector.h`/`HandDetector.h`.
- `ClassRegistry` is the one place class IDs come from: the 80 COCO classes (put `coco.names` back to 80 lines), built-in `face` and `hand`, and custom labels added at runtime. Name lookups are bounds-checked, which fixes the out-of-bounds read at `Logger.cpp:30`.
- An `IDetector` interface implemented by `YoloDetector`, `FaceDetector` and `HandDetector`. A `DetectorSet` (built as `Detector`) runs the enabled ones and applies the class filter.
- **Remove `personByFace` completely:** `person` is the YOLO body box, `face` is its own type, `hand` is its own type.
- The palm box is enlarged to a hand box. It's expanded about 2.5× toward the fingers, using the wrist → middle-finger keypoints the palm model already outputs (its 18 outputs per box are 4 box values plus 7 keypoints). This is the same approach as MediaPipe.
- Thresholds per detector: `--conf-yolo 0.4`, `--conf-face 0.6`, `--conf-hand 0.6`; `--conf` sets all three. Each detector also returns low-score detections (≥0.1) for M3's second matching pass.
- Model load failure is a hard error if the model was asked for with a flag. A default model that fails to load prints a warning and is disabled.

### M2: Frame rate (target ≥25 FPS live; the camera's maximum is 30)
**Done 2026-10-17.** Realtime playback of `hands_face.mov` holds 30.0 FPS with no skipped frames; processing speed went from 18.4 to 74.8 FPS (detection about 13 ms per frame). As built:
- YOLO 640×384 runs fully on CoreML (MLProgram, 233/233 nodes): 44 ms on the CPU → 11 ms.
- The palm model runs on the CPU. CoreML could only take 91 of its 124 nodes, and that split ran slower (14.6 ms) than CPU alone (12.5 ms). The wrapper (`OrtModel`) now uses CoreML only when it takes a whole model.
- Faces run on OpenCV at 640 px wide (7 ms).
- Detectors side by side: 29 ms → 13 ms per frame.
- `--detect-every N` was not needed (detection uses well under half the frame budget), so it was left out. `--stats` and `--benchmark` became an end-of-run summary printed by every run, plus `--realtime` and `--sequential`.

Original plan:
- **Concept: CoreML execution provider.** ONNX Runtime can hand a model to Apple's CoreML, which runs it on the M1 Pro's GPU or Neural Engine instead of the CPU.
- `brew install onnxruntime` (the formula builds with CoreML on macOS; header `onnxruntime/coreml_provider_factory.h`). An `OrtSession` wrapper (built as `OrtModel`) tries CoreML first and falls back to CPU, and prints the backend it picked. CoreML's compiled model is cached so restarts are fast. Flag: `--backend coreml|cpu`.
- YOLO and the palm model move to ONNX Runtime. YuNet stays on OpenCV's `FaceDetectorYN` but runs on a 640×360 downscale, and boxes are scaled back up.
- YOLO is re-exported at **640×384** instead of 640×640. A 16:9 frame then wastes almost no compute on padding (about 40% less work). The decoder reads the input size from the model.
- **Capture thread:** it always holds the newest frame, so camera waits never block processing.
- **Detectors run in parallel** (`std::async`, joined each frame), so time per frame is the slowest detector rather than the sum of all three. `cv::setNumThreads` is tuned so the threads don't compete for cores.
- Fallback flag `--detect-every N`: frames in between use Kalman predictions only.
- HUD and `--stats` show milliseconds per stage and FPS. `--benchmark` runs a clip and prints averages. (As built: the HUD shows FPS, and the detector and drawing times only in the debug view, `d`. The summary every run prints at the end has the time per stage.)

### M3: Tracker rebuild (stable IDs, numbers per type)
**Done 2026-10-17.** Confirmed IDs with the full program (same clips as the baseline):

| | Baseline | M3 |
|---|---|---|
| `hands_face` hand IDs | 75 | 4 (two hands keep #1 and #2 for the first 47 s) |
| `hands_face` face IDs | 7 ("person") | 1 |
| `flashlight` hand IDs | 25 | 4 |
| `flashlight` cell phone IDs | 10 | 4–6 (CoreML's GPU math varies a little between runs) |

What the detection data showed, and what fixed it:
- Hands drop out for 1–10 frames about 90 times per minute. Velocity now fades while a track is unseen, so the track waits nearby instead of drifting off.
- Hands often leave through the bottom edge. Edge-clipped boxes keep the track's real size, and the track waits up to 3 s at the edge (`exitTimeout`).
- Hand boxes change size up to 3× between a fist and an open hand. The size limit is now 3×, and distance is measured against the geometric mean of the two sizes.
- The palm model sometimes reports weak, frame-sized junk boxes. Weak detections must overlap the track, and a much bigger box never matches.
- A phone-screen "person" absorbed nearby objects. A track may change class only when the boxes overlap ≥ 0.5 (same object relabeled).
- A false "toilet" lived on weak detections. A YOLO object seen only weakly for over 1 s is hidden, and any track without a confident detection for 5 s is dropped.

`tools/replay.cpp` (`tracker_replay`) re-runs the tracker on saved detections in a fraction of a second, for tuning. A parameter sweep showed the results are stable across wide ranges.

Known limitation: hands have no appearance cue, so when both are hidden at once they can swap numbers when they come back.

Original plan:
- **Concepts:**
  - *Hungarian algorithm*: finds the best overall one-to-one pairing of tracks and detections, instead of greedily taking the best pair first.
  - *ByteTrack*: a second matching pass with low-confidence detections, so a blurry hand still updates its track instead of losing it.
  - *DIoU*: box overlap minus a penalty for center distance, so fast-moving small boxes still match.
- `Assignment.{h,cpp}`: Hungarian matching, unit tested.
- Matching order:
  1. Confident detections against active tracks.
  2. Low-score detections against the tracks still unmatched (these can't start new tracks).
  3. Lost tracks re-acquired with a wider search window around their last position.
- Tracks only match within the same kind (object, face or hand). COCO labels may differ at a cost penalty, and a track's class is decided by vote over recent frames. A tv/laptop flicker no longer breaks a track.
- Track states: Tentative → Confirmed → Lost → Deleted.
  - While lost, velocity decays with a 0.15 s half-life so the box stops drifting away. It's based on time, not frames, so it behaves the same at any FPS.
  - Timeouts are in **seconds** (lost after 0.7 s), not frames, so behavior doesn't change with FPS.
  - The Kalman filter steps by the real time between frames, using capture timestamps. Width and height velocities are clamped.
- **Numbers per type** (basic version done early, 2026-10-17: per-class counters assigned on confirmation, never reused; M3 extends it to class changes and custom-label types): each type has its own counter (`hand #1`, `hand #2`, `face #1`, `cup #1`).
  - A number is assigned **when a track is confirmed**, so 1–3-frame noise tracks don't use up numbers.
  - Numbers are never reused within a run.
  - If a track's voted class changes for good (e.g. it settles on `laptop` after starting as `tv`), it gets the next number in the new type. The old number is not handed out again.
  - `uid` (built as `Track::id`, the log's `id` column) stays unique for the whole run and goes into the logs.
- `Track` becomes move-only, which removes the trap where a copy shares the Kalman filter's matrices. Confidence is smoothed before display.

```cpp
enum class TrackState { Tentative, Confirmed, Lost };
struct Track {
    int uid;                 // unique for the whole run (logs)
    int number = 0;          // per-type number shown on screen; assigned on confirmation
    ClassId cls;             // decided by vote over recent hits
    TrackState state;
    KalmanBox kf;            // dt-aware constant-velocity filter
    float conf;              // smoothed (EMA)
    double lastSeen;         // seconds, capture time
    TrackIdentity identity;  // custom-label votes (M5)
    bool target = false;     // M4
    std::deque<cv::Point2f> trail;
    Track(const Track&) = delete;
    Track(Track&&) = default;
};
```
As built (`src/Tracker.h`): `id` and `classId` instead of `uid` and `cls`, and flags (`confirmed`, `visible`, `matched`, `coasting`) instead of `TrackState`. Label votes and the target are kept in `Recognizer` and `TargetSelector`, not on the track.

### M4: Bright-light face targeting
**Done 2026-10-17.** On `flashlight.mov`, all 10 times the light hit the face became TARGET episodes. Each turned on within 0.2 s and released about 0.25 s after the light went off. The phone (the light source) and the chair were never targeted. On `hands_face.mov` there were no targets. Measured and changed from the plan:
- In a bright room an unlit face measured 0.72–0.78 of the room's brightness, and a lit one 1.2–1.8. A rise of 1.5–2.6× over normal is the reliable signal. Thresholds: rise ≥ 1.35 and face at least as bright as the room (ratio ≥ 1.0); the planned ratio ≥ 1.6 missed half the episodes.
- After the light goes off, the face dips to about 0.5 for a second while the camera's exposure recovers. The normal level now adapts quickly only within ±15% of normal, and very slowly (20 s) otherwise, so lit periods and dips don't drag it along.
- A hand held in front of the face (a palm lit by the window) looked like a light: rise 1.55. Faces covered more than 25% by a tracked hand, or not detected that frame, aren't measured.
- Track colors skip red, so red only ever means TARGET.

Original plan:
**Rule (user, 2026-10-17): only a change in brightness on a face triggers TARGET.**
- **Only faces can be targeted.** Brightness is measured only on face tracks (and custom labels whose base class is face). The phone shining the light, a lamp, a window or a screen is never targeted, even if it's the brightest thing in the frame. In `flashlight.mov` the phone is detected as `cell phone`, which is correct, and it must stay an ordinary track.
- `TargetSelector` runs after the tracker and measures, for each confirmed face track:
  - **Face brightness:** average luma over the middle 60% of the face box, to avoid hair and background.
  - **Background brightness:** median luma of a 160×90 downscale of the frame, with every face box masked out. A median barely moves for a small, very bright spot, such as the flashlight LED itself or a lamp in view.
  - **Ratio** = face ÷ background, so brightness is judged relative to room lighting.
  - **Clipping:** if more than 40% of face pixels are clipped near 255, the face counts as very bright. Webcam auto-exposure tends to clip a face in a flashlight beam.
- **The trigger is a change on the face.** Each face has a *normal ratio*: its usual brightness relative to the room. A face becomes a candidate only when **both** of these hold:
  1. Its ratio has **risen to ≥1.5× its own normal ratio**. A light has just started shining on it.
  2. Its ratio is **≥1.6**: it's now clearly brighter than the background.

  A face that's simply brighter than its surroundings all the time (monitor glow in a dark room, a sunny spot) never triggers, because nothing changed on it.
  - Ratios are compared rather than raw brightness. When a light hits the face, webcam auto-exposure darkens the whole image, so raw face brightness may barely rise even though the face-to-background ratio jumps.
  - The normal ratio is a slow running average (about 3 s), updated only while the face isn't rising or targeted.
  - A newly seen face needs 1 s of history before it can be targeted. So a face that arrives already lit isn't targeted until the light on it changes.
  - A room light switching on or off changes face and background together, so the ratio, and therefore the trigger, stays put.
- **Concept: hysteresis.** Different thresholds to turn on and turn off, so the state doesn't flicker:
  - Target when both conditions hold for 0.1 s.
  - Release when the rise drops below 1.25× normal for 0.25 s, for example when the light goes off.
  - Both are timed in seconds rather than frames. Thresholds are tuned on `flashlight.mov`.
- **Brightest only:** of the faces that qualify, the highest ratio wins. The current target keeps it unless another face beats it by 0.2.
- **Overexposure:** a strongly lit face can stop being detected. The target box then keeps following the Kalman prediction for up to 0.5 s while that region stays bright. (In the baseline, the face track survived the whole flashlight clip, so this is a safety net.)
- **Display:** the box is red and 3 px thick, and the label is exactly `TARGET`. Normal track colors **skip red hues** (`trackColor` in `Overlay.cpp`, called `colorForId` before the UI polish round), so red only ever means TARGET.
- Flags: `--no-targeting`, `--target-ratio` (default 1.6), `--target-rise` (default 1.5).

```cpp
struct FaceLight   { float normalRatio = 0; float ratio = 0; double firstSeen = 0; };  // per face track uid
struct TargetState { int trackUid = -1; double conditionSince = 0, releaseSince = 0, coastUntil = 0; };
```
As built (`src/Targeting.h`): `FaceLight` is kept per face track `id`, with more fields, and `Target` (track id, box, ratio, rise) takes the place of `TargetState`. The timers are members of `TargetSelector`.

### M5: Custom labels
**Done 2026-10-17.** Tested with frames from the clips as photos (kept outside the project):

| Label | Photos from | Recognized in `flashlight.mov` |
|---|---|---|
| `me` (face, SFace) | `hands_face.mov` | 99% of frames. Every check matched (0.51–0.87 vs 0.36 needed). |
| `my_phone` (cell phone, DINOv2) | `flashlight.mov`, 3 usable photos | 75% of frames: every phone track matched at 0.62–0.91. The rest is the first second of each track, while it gathers 3 matching checks. |

Measured and changed from the plan:
- DINOv2 summary token vs. the average of its patch tokens: on 405 phone crops vs. other objects, patch averaging with a tight crop separated them best. Same phone: 0.77 typical, 0.57 at the 10th percentile. Other objects: 0.29 typical, 0.57 at the 90th. AUC 0.965, against 0.937 for the summary token. Threshold 0.6.
- DINOv2 runs on the CPU (84 ms per crop on the recognition thread). CoreML ran it in 8 ms but took 8–38 s to load on every launch. With the original choice, enrolling 5 photos took 46 s; it now takes 2 s.
- Enrollment skips photos where the item isn't found (`--whole-photo` for close-ups), and picks the face rather than the whole `person` when the photos show a face.
- Unlabeled tracks are checked every 0.25 s and labeled ones every 1 s. 4 misses in a row take a label away.
- Not tested: telling two different items of the same class apart (the clips have one phone and one person). The 0.6 threshold sits above 90% of other-object scores. Identical products can't be told apart from pictures.

**Label database, 2026-10-20** (branch `v2-labels-db`): each label is a folder of photos, synced at startup. Embeddings are cached per photo in `.embeddings.yml` (keyed by size and modification time), and `label.yml` now holds optional settings (`class`, `whole_photo`, `threshold`). Also added: `--sync-labels`, `--add-photos`, `--rename-label`; `--remove-label` moves the folder to `labels/.removed/`. A sync warns about unusable photos, labels with few usable photos, outlier photos and look-alike labels. Old labels are converted on load. Checked with labels rebuilt from the clips in the old format: after conversion, `me` was still on 99% of face frames and `my_phone` on 75% of phone frames. Drop-folder labels (`tools/grab_frames` photos from the clips) got 99% (face) and 76% (phone, close-up crops with `whole_photo`).

Original plan:
- **Concept: embeddings.** A neural network turns an image crop into a vector, and crops of the same item produce vectors pointing in nearly the same direction. Cosine similarity measures that, and several photos give several vectors, which covers more angles and lighting.
  - **Faces:** SFace (OpenCV's `FaceRecognizerSF`, `face_recognition_sface_2021dec.onnx`), aligned using YuNet's landmarks. It's built for telling faces apart.
  - **Objects:** DINOv2-small (pre-exported ONNX, run on ONNX Runtime with CoreML), which is strong at recognizing specific instances.
- Structs:
```cpp
using Embedding = std::vector<float>;        // L2-normalized → dot product = cosine similarity
struct LabelSample   { fs::path image; cv::Rect roi; Embedding embedding; };
struct CustomLabel   { std::string name; ClassId baseClass;     // e.g. "my_mug" refines "cup"
                       std::vector<LabelSample> samples;        // multiple photos per item
                       float threshold; };
struct LabelMatch    { int label = -1; float score = 0, margin = 0; };
struct TrackIdentity { int label = -1; std::map<int, float> votes; double lastEmbedAt = 0; };
class  LabelLibrary  { load(dir); enroll(...); match(baseClass, emb) -> LabelMatch; save(); };
```
- **Enrolling:** `tracker --enroll my_mug a.jpg b.jpg c.jpg [--class cup]`
  1. Copies the photos to `labels/my_mug/`.
  2. Detects the item in each photo: the largest box of `--class`, or the base class inferred across photos. If nothing is found it uses the whole image and warns.
  3. Embeds each crop and saves `label.yml` with OpenCV's `FileStorage`, tagged with the embedding model version so a stale cache is recomputed.
  4. Also stores an embedding of each crop **flipped left-to-right**. The live camera image is mirrored but photos aren't, so a mug's logo appears reversed on screen. Storing both directions keeps recognition from depending on the mirror setting.
  5. Reports how consistent the samples are and flags photos that don't fit. Five to ten photos from different angles and lighting are recommended.
- Other commands: `--list-labels`, `--remove-label`, `--reembed`. (`--reembed` wasn't built. With the label database, a sync redoes a label's photos when the embedding model or its settings change.)
- **At runtime:**
  - A confirmed track of a base class is embedded on the worker thread, at most 2 crops per frame. Each track is re-checked about every 0.5 s or when its size changes a lot.
  - A label's score is the average of its top-3 sample similarities. The best label needs score ≥ threshold **and** a 0.05 lead over the runner-up.
  - Votes across frames must agree 3 times before a track gets the label, and the label stays put once given.
  - Each label is one specific item, so only one track can hold it at a time.
- **Display:** `my_mug`, with no number. A label is one specific item, so a number would always be #1. The `cup #N` number the track had before is not handed out again.

### M6: Polish
**Done 2026-10-17.** Most of it landed during M1–M5:
- checked flag values, trimmed `--classes`, removed redundant flags;
- options in `Options.cpp`, unit tested;
- log with frame time, `matched`, `label`, `target` and `light_ratio`, flushed every second;
- clean shutdown on Ctrl-C;
- `--realtime`, the actual camera resolution printed at startup;
- label text black or white for contrast, HUD timings per detector (both replaced in the UI polish round, below);
- keys `p` and `t`.

Added in M6:
- `--record PATH`: the raw, unmirrored feed, which plays back with `--video`.
- The debug view (`--debug`, key `d`): each face's light ratio and rise, and each track's closest label and score.
- Predicted boxes drawn thinner.
- A hand blocks a face's light measurement only if it was confidently detected in the last 0.5 s. The debug view showed a weak phantom "hand" on the face blocking it.
- Hands and faces seen only weakly for over 2 s are hidden. They keep their IDs until 5 s without a confident detection.

Not done: `--models-dir` (the default paths suffice). `Track` stayed copyable instead of move-only, because the Kalman filter is now a plain value type (`KalmanBox`), so copies no longer share state.

Original plan:
- **CLI:** checked numbers with clear errors, ranges enforced, spaces trimmed in `--classes`, redundant flags removed (`--person-by-face`, `--enable-hands`, `--no-trail`), `--models-dir`.
- **Logger:** new columns `t_capture_ms,frame,uid,type,number,class,label,state,x,y,w,h,conf,target,light_ratio`. (As built: `time_ms,frame,id,number,class,label,matched,x,y,w,h,conf,target,light_ratio,coasting`; `coasting` came with the lag and flicker round.) Timestamps are capture time. It flushes every second, and Ctrl-C shuts down cleanly.
- **Video:** `--realtime` plays at the file's own FPS. `--record out.mp4` saves the raw camera feed for making new test clips.
- **Overlay:** label text in black or white depending on the box color. HUD timings per stage. Keys: `p` pause, `t` trails, `d` debug overlay (luma ratios, match scores, lost tracks shown dashed), `q`/ESC quit. (Replaced in the UI polish round, below: white text on a dark tag with a stripe of the track's color, detector and drawing times only in the debug view, predicted boxes drawn as corners only, and `h` for help. Dashed lost tracks weren't built.)
- **Camera:** checks the resolution it actually got and prints it.
- Comments cleaned up (no change-log wording, no unmeasured claims). README updated.

### Lag and flicker (2026-10-20, branch `v2-tracking`)
On the live camera, boxes trailed behind moving hands and flickered. Measured on the reference clips (lag on
replays of the same detections, the rest on full runs), hands unless noted, `hands_face` / `flashlight`:

| | Before | After |
|---|---|---|
| Fast motion: box behind its detection, mean / 90th pct (% of box size) | 8.8 / 15.4, 11.4 / 22.6 | 4.0 / 7.0, 4.4 / 7.9 |
| ... how far behind in time | 13 ms | 6–7 ms |
| Not detected: box off the hand's path, mean / 90th pct | 33 / 71, 34 / 89 | 29 / 61, 21 / 36 |
| Box switches to thin, per minute | 57, 15 | 24, 11 |
| Box vanishes and comes back, per minute | 15.1, 4.6 | 13.3, 2.3 |
| Wobble while slow, 90th pct (% of box size) | 4.7, 6.1 | 4.2, 5.3 |
| Phone: class / number changes per minute | 5.7 / 3.4 | 2.3 / 2.3 |
| Hand IDs, realtime FPS, labels | 4, 4; 30; face 99%, phone 75–78% | unchanged |

What the measurements showed:
- The pipeline wasn't the lag: capture to screen takes about 24 ms (detection 14 ms, tracking under 0.1 ms),
  and boxes are drawn on the frame they were detected in. Every run now prints this breakdown.
- The Kalman filter trailed a waved hand: it keeps about 40% of its old estimate each frame. Where the filtered box
  still differs from the detection by more than the detection jitter, it now snaps toward the detection (fully
  at twice the jitter; width and height at twice that). Weak detections count as 1.5× noisier: snapping
  to every noisy weak palm box made the next one miss the track.
- A missed hand was predicted worse than simply holding its last box (it turns or changes shape rather than
  flying on), so velocity now halves every 0.05 s while unseen instead of 0.15 s.
- Most thin boxes were 1–2 frame detector misses. A box is drawn thin only after 0.1 s without a detection.
- Weak detections revived lost tracks and dragged them onto junk (with snapping, a false toothbrush stayed up
  for 20 s). They now only keep a track going that was seen in the last 0.5 s.
- The flashlight phone was called chair, tv, chair, vase, cell phone, with a new number each time. Class votes
  fade more slowly (0.95 per match), and a track that returns to a class keeps its number in it.
- A hand passing in front of the lit face blocks its light measurement for over 0.5 s, which dropped TARGET for
  a few frames (twice before, three times with the faster hand boxes). The target now rides out 0.75 s. All
  lit periods still become TARGET at the same frames; there are 8 unbroken TARGET runs where there were 10
  pieces before, and one releases 0.3 s sooner, right after the light goes off.
- Left as is: the coast time (0.2 s). Most remaining gaps are hands gone for over half a second, and a longer
  coast only kept ghost boxes up longer.

### UI polish (2026-10-20, branch `v2-ui`)
The overlay was rebuilt to read at a glance: tags stay inside the frame and off each other, boxes don't flicker between widths, the TARGET stands out, and debugging numbers moved to the debug view. Drawing time per frame, from the run summary's `Drawing per frame` line (added in this round):

| | Before | After |
|---|---|---|
| Drawing per frame | 0.13 ms | about 0.2 ms |
| ... with the debug view and trails | 0.3 ms | about 0.5 ms |

What changed:
- `Overlay` (`src/Overlay.{h,cpp}`) replaces `drawTracks`, `drawHud` and `HudInfo`. `Overlay::draw` takes the tracks, `DrawOptions` (trails, debug, help) and a `FrameInfo` (FPS, detector and drawing times, paused, recording), and keeps each tag's place from frame to frame.
- Text uses OpenCV 5's built-in sans-serif font, at a size that scales with the frame height.
- Tags: the name in white on a dark translucent plate with a stripe of the track's color (`trackColor`, still never red). This replaces the black-or-white text on a box-colored label. Confidence moved to the debug view, whose details (confidence, light ratio and rise, closest label and score) are extra lines in each tag.
- `TagLayout` (`src/TagLayout.{h,cpp}`) places the tags. Each tries spots around its box, best first: above the top-left corner, just inside it, the same at the top-right corner, then the bottom corners and further out. Spots are moved inside the frame and, while there is room, kept off earlier tags, the HUD and the TARGET LOCKED badge. The TARGET's tag is placed first, then custom labels', then the oldest tracks'. A tag pushed to a worse spot moves back only after 15 frames there, and only if the better spot has room to spare: quick back-and-forth hops on `hands_face.mov` went from 26 to 2.
- Boxes: a thin outline with heavier corners. A predicted box (`Track::coasting`: no detection for 0.1 s; the lag and flicker round counts these as "thin") shows only its corners, so one or two missed frames don't change the look.
- TARGET: a thick red box with pulsing corner marks around it, a red `TARGET` tag, and a "TARGET LOCKED" badge at the top center.
- HUD: a translucent panel in the top-left corner with FPS, chips for PAUSED, REC, DEBUG and TRAILS, and how many of each class are tracked. Detector and drawing times show only in the debug view (`d`).
- `h` shows a help panel with the keys and what the red and corner-only boxes mean. A "Press h for help" hint fades out after a few seconds.
- Trails (`t`) fade and thin with age, and skip the jump when a lost track is found again.
- While paused (`p`), the frame is redrawn, so PAUSED shows and `t`, `d` and `h` still work. The latency summary leaves out time spent paused.
- Translucent shapes blend only the pixels they cover.
- Unit tests: `tests/OverlayTest.cpp` and `tests/TagLayoutTest.cpp`.

Not checked: no live-window run of this round is recorded. The keys, the paused redraw, the help panel and hint, the pulsing TARGET corners and how the tags read on the live camera are unverified. The commit doesn't say which clip the drawing times come from.

## Risks and fallbacks
- **CoreML support:** checked 2026-10-15. Homebrew's ONNX Runtime (1.31.0) is built with `-Donnxruntime_USE_COREML=ON` on macOS. `Ort::GetAvailableProviders()` confirms it at startup anyway.
- **CoreML may not run every layer.** Unsupported layers (more likely in DINOv2's transformer) fall back to the CPU. The startup list of where each detector runs, the detector times in the debug view (`d`) and the end-of-run summary show it, and `--backend cpu` can be used for comparison. (As built, a model runs on CoreML only if CoreML takes all of it; see M2.)
- **The YOLO re-export needs Python `ultralytics`** in `.venv`. If that's not wanted, YOLO stays at 640×640 with some speed lost.
- **Model sources:** checked 2026-10-15. DINOv2-small ONNX is at Hugging Face `onnx-community/dinov2-small` (`onnx/model.onnx`, 88.5 MB), and SFace is in `opencv_zoo`. The fallback for DINOv2 is MobileNetV3.
- **Webcam auto-exposure** shifts brightness. The ratio-based design handles most of it, and the thresholds will be tuned on `flashlight.mp4`.
- **CoreML's first model compile** can take seconds. The compiled-model cache directory avoids repeating it.

## Baseline (M0, measured 2026-10-17)
Original tracker on the reference clips (`--video`, default flags, so "person" is the face box). FPS is processing speed on video; live camera runs were about 10 FPS.

| Clip | Length | FPS | Confirmed IDs | Hand IDs (short ≤5 frames) | Person/face IDs | Other |
|---|---|---|---|---|---|---|
| `hands_face.mov` | 2026 frames, 67.6 s | 14.2 | 83 | 75 (25) | 7 | toothbrush 1 |
| `flashlight.mov` | 1576 frames, 52.6 s | 14.3 | 39 | 25 (11) | 1 | cell phone 10, chair 2, toothbrush 1 |

Logs: `logs/run_20261017_182032.csv`, `logs/run_20261017_182256.csv` (not in git; numbers above are the record).

## Verification
- **Unit tests (`ctest`):**
  - Hungarian matching and DIoU.
  - Tracker scenarios: straight-line motion → 1 ID; 0.5 s occlusion → same ID; two crossing objects keep their IDs; label flicker → still 1 track; noise tracks don't use up numbers; `hand #1/#2` and `face #1` are numbered separately.
  - TargetSelector on synthetic frames:
    - A face that brightens → target. A face that's dim, or bright all along with no change → none.
    - Hysteresis and brightest-only.
    - A bright non-face box (a phone) → never a target.
    - The whole frame brightening (room light on) → no target.
    - A new face with under 1 s of history → no target.
  - LabelLibrary save/load round trip and the lead rule; ClassRegistry; CLI parsing.
- **Regression clips** (run with `--video ... --stats`, compared to the M0 baseline; as built, every run prints the summary and there is no `--stats`):
  - `hands_face.mov`: total confirmed IDs down ≥ 75% from the M0 baseline on the same clip, at most 2 IDs per hand that's in view, and face and hand counted as separate types. For reference, an earlier live run gave 2 hands 18 IDs.
  - `flashlight.mov`: TARGET appears within 0.2 s of light-on and clears within 0.5 s of light-off. The `cell phone` track (the light source) and the chair are never targeted.
- **Live checks:**
  - ≥ 25 FPS with all detectors on, **and still ≥ 25 FPS with custom labels enrolled** (recognition shares the GPU). The startup list shows YOLO on CoreML (the HUD doesn't show the backend).
  - Turning room lights on and off produces no TARGET.
  - After enrolling a mug from 5 photos, it shows as `my_mug` while a different cup stays `cup #N`. This holds with and without `--no-mirror`.
