# Object Tracker

Tracks people, faces and hands through the MacBook camera (or a video file) and draws a live overlay with
an ID per track, and a red `TARGET` box on a face lit by a bright light. Each run also writes a CSV log of
every visible track.

- **Detection:** YOLOv8n for people (the model has COCO's 80 classes; only person's scores are read),
  YuNet for faces, MediaPipe palm detection for hands. The three run side by side; YOLO runs on the GPU /
  Neural Engine through ONNX Runtime + CoreML.
- **Tracking:** one Kalman filter per track in real time units; optimal (Hungarian) matching on box overlap
  and center distance; confident detections first, then weak ones to keep existing tracks going (ByteTrack's
  idea). Tracks ride out short drop-outs and wait at the frame edge for objects that left. Each type is
  numbered on its own (`hand #1`, `hand #2`, `face #1`), and numbers are never reused within a run.
  The filter smooths detection jitter, but a box snaps to its detection when the object moves faster than
  the filter expects (a waved hand), so it doesn't trail behind.
- **TARGET:** when a bright light suddenly shines on a face, that face gets a red box labeled `TARGET`.
  Brightness is judged relative to the room and to the face's own normal level, so a face that is always
  bright, a room light switching on, a bright phone or a hand held up to the face don't count. Only faces
  can be targets, and only the brightest lit one at a time (`--no-targeting` turns it off).

`v3` is this people-only version. `v2` has every feature (all 80 COCO classes, object labels), and was
built following [PLAN.md](PLAN.md); `main` holds the original version.

## Setup (macOS, Apple Silicon)
```sh
brew install cmake opencv onnxruntime googletest clang-format
scripts/fetch_models.sh          # downloads and checksums the face, hand and recognition models
scripts/export_yolo.sh 384 640   # YOLO with a 16:9-friendly input (installs Ultralytics in .venv)
```

## Build and test
```sh
cmake -S . -B build
cmake --build build -j8
ctest --test-dir build           # unit tests (needs googletest)
```

## Run
```sh
./build/tracker                          # camera 0: people, faces and hands
./build/tracker --classes face,hand      # only these classes (person, face, hand, label names)
./build/tracker --video clips/hands_face.mov
./build/tracker --video clips/hands_face.mov --no-window   # headless: just the log and a summary
./build/tracker --video clips/hands_face.mov --realtime    # play at camera speed, skipping frames when busy
./build/tracker --backend cpu                             # don't use CoreML
./build/tracker --debug                                   # show confidences, light ratios, label scores and timings
./build/tracker --record clips/new.mov                    # also save the raw camera feed
./build/tracker --conf-hand 0.5                           # per-detector thresholds: --conf-yolo/-face/-hand
./build/tracker --list-classes
./build/tracker --help                   # all options
```
Keys: `p` pause/resume, `t` trails on/off, `d` debug view on/off, `h` help, `q` or `Esc` quit. Ctrl-C in the terminal also stops cleanly.
The panel in the top-left corner shows the frame rate, how many of each class are tracked, and which modes are on.
Boxes drawn with only their corners are predictions: the object hasn't been detected for a few frames, and the tracker is keeping its box going.
The first camera run asks for camera access: allow it for your terminal app in **System Settings → Privacy & Security → Camera**.

At startup it prints where each detector runs. CoreML compiles a model on first use (a few seconds) and
keeps the result in `.cache/coreml/`. A model runs on CoreML only if CoreML can take all of it; otherwise
it runs on the CPU, which was faster for the partly supported palm model.

## Custom labels (faces)
Teach it a specific person's face from a few photos, so their face shows their name instead of `face #n`. Each label is a folder of photos in `labels/`:

1. Make a folder named after the person, e.g. `labels/alex/` (letters, digits, `_` and `-`; not `TARGET`, which is the red box's tag).
2. Put **5–10 photos** of their face there, from different angles and in different light. JPEG and PNG work; convert iPhone HEIC photos first (`sips -s format jpeg IMG_1234.heic --out IMG_1234.jpg`).
3. Run the tracker. At startup it looks at new, changed and removed photos, says for each one whether it found a face, and warns about anything that would hurt recognition.

```sh
./build/tracker --sync-labels                  # do step 3 now, and check every label
./build/tracker --list-labels                  # photos and usable photos per label
./build/tracker --enroll alex a.jpg b.jpg      # or: copy photos into labels/alex/ and sync
./build/tracker --add-photos alex c.jpg        # copy more photos in
./build/tracker --rename-label alex alexis
./build/tracker --remove-label alexis          # moves labels/alexis/ to labels/.removed/, photos and all
./build/tracker --classes alex                 # show only alex's face
./build/tracker --no-auto-photos               # recognize, but take no photos (see below)
./build/grab_frames clips/new.mov /tmp/alex --crop face --count 8   # face photos from a video
```
- It stores a neural-network *embedding* (SFace) of the face in each photo, plus a mirrored copy, in the folder's hidden `.embeddings.yml`. Only new or changed photos are looked at again. An optional `label.yml` can set `threshold:` (how close a face must be to count) and `auto_photos: false`.
- Photos without a face aren't used. A sync also points out photos that don't look like the rest, labels with fewer than 5 usable photos, and labels so alike that recognition may mix them up.
- A sync needs only the face detector. If it didn't load, startup leaves the labels as they were and says so, and `--sync-labels`, `--enroll` and `--add-photos` stop with an error. With `--no-faces`, startup leaves new photos for `--sync-labels`.
- While tracking, faces are checked in the background a few times a second. A face takes the label after 3 matching checks, and each label is one person, so at most one face has it. Label changes are picked up at the next start.
- **It learns as it goes.** While it recognizes someone, it now and then takes a photo of their face for their label, e.g. `labels/alex/auto_20261104_153012.jpg`: at most one every 30 s, and only of a face recognized clearly and steadily (5 checks in a row), seen well (detected, at least 80 px wide, sharp, not under the TARGET's light) and from an angle or in light the label's photos don't show yet. It's used at once, and from the next start like any other photo. A label keeps up to 40 of them; then the one most like another photo makes way (your own photos are never touched). A sync lists them: delete any that show someone else. `--no-auto-photos` turns this off for a run, `auto_photos: false` in `label.yml` for one label.
- **Face network.** A small neural network (`labels/.face_net.yml`) learns the labeled faces from all their photos together, and what other people look like, from faces it saw that matched no label. It's retrained in the background after new photos, and by `--sync-labels`. It needs 3 photos of each label and 20 faces of at least 5 other people (seen while tracking); until then it's off and recognition uses the photos alone, as the startup line, `--sync-labels` and `--list-labels` say. Once on, a face scoring up to 0.08 below a label's threshold still counts if the network is at least 90% sure it's that person, and a match doesn't count if it's 90% sure it's someone else. The debug view shows its answer for each face.
- **Privacy:** all of it stays on this computer, in `labels/`. Photos are only taken of people you labeled, while they're recognized. Faces of anyone else are never saved as photos: for the network it keeps only their embeddings (128 numbers per face, at most 500, no names) in `labels/.negatives.yml`; delete that file to forget them. With `--no-auto-photos` it keeps neither.
- A face under a bright light still shows `TARGET`.
- **Labels from `v2`:** face labels work as they are. Object labels (e.g. a phone) are reported as unsupported and left untouched, so they still work if you switch back to `v2`; `class:` and `whole_photo:` lines in `label.yml` are ignored. Labels made by an even earlier version (one `label.yml` with the embeddings) are converted automatically; the old file is kept as `.label-v1.yml`.
- `labels/` holds your photos and isn't committed to git.

## Logs
Each run writes `logs/run_<date>_<time>.csv`: one row per visible track per frame. The columns are `time_ms`, `frame`, `id` (unique for the run), `number` (the one on screen), `class`, `label`, `matched` (0 = not detected this frame), the box, `conf`, `target`, `light_ratio` (faces) and `coasting` (1 = drawn with corners only, as a prediction). To summarize one or more runs:
```sh
scripts/log_stats.py logs/run_*.csv
```
It prints the frame rate, the number of IDs per class and how long tracks lasted. Fewer IDs and longer lifetimes mean steadier tracking. It also counts flicker per minute: boxes that vanish and come back, switch to thin, wobble or change class, label or number.

Every run ends with the time per frame of each stage (mean and 95th percentile): how long the frame waited in the camera, detection, tracking, labels and targeting, drawing, display, and capture to screen in total.

To tune the tracker without re-running the neural networks, save a clip's detections once and replay them:
```sh
./build/tracker --video clips/hands_face.mov --no-window --no-log --dump-detections /tmp/dets.csv
./build/tracker_replay /tmp/dets.csv --tracks lostTimeout=1.5   # any TrackerParams field
```
Besides IDs it reports flicker and lag per class: how far the shown box is from its detection, overall and while the object moves fast, and how far behind it trails.

## Layout
```
src/        Camera (capture thread); ClassRegistry (person, hand, face, labels); Detector runs the
            IDetector implementations (YoloDetector, FaceDetector, HandDetector) side by side;
            OrtModel (ONNX Runtime + CoreML); Tracker (+ KalmanBox, Assignment); Targeting (TARGET);
            Embedder, Labels, Enrollment, Recognizer (face labels); Options; Overlay
            (+ TagLayout); Logger; main
labels/     face labels: a folder of photos per person, with optional label.yml (not in git)
tools/      replay.cpp: the tracker on saved detections (tracker_replay); grab_frames.cpp:
            a video's frames, or face crops, saved as photos (grab_frames)
tests/      GoogleTest unit tests (tracker_tests)
scripts/    fetch_models.sh, export_yolo.sh, log_stats.py
models/     *.onnx (downloaded, not in git)
clips/      reference videos for regression runs (not in git)
logs/       run logs (not in git)
```
