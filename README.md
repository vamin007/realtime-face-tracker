# Object Tracker

Tracks objects through the MacBook camera (or a video file) and draws a live overlay with an ID per object.
Each run also writes a CSV log of every visible track.

- **Detection:** YOLOv8n for the 80 COCO classes, YuNet for faces, MediaPipe palm detection for hands.
  The three run side by side; YOLO runs on the GPU / Neural Engine through ONNX Runtime + CoreML.
- **Tracking:** one Kalman filter per track in real time units; optimal (Hungarian) matching on box overlap
  and center distance; confident detections first, then weak ones to keep existing tracks going (ByteTrack's
  idea). Tracks ride out short drop-outs and wait at the frame edge for objects that left. Each type is
  numbered on its own (`hand #1`, `hand #2`, `face #1`), and numbers are never reused within a run.
- **TARGET:** when a bright light suddenly shines on a face, that face gets a red box labeled `TARGET`.
  Brightness is judged relative to the room and to the face's own normal level, so a face that is always
  bright, a room light switching on, a bright phone or a hand held up to the face don't count. Only faces
  can be targets, and only the brightest lit one at a time (`--no-targeting` turns it off).

The `v2` branch is being rebuilt following [PLAN.md](PLAN.md). `main` holds the original version.

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
./build/tracker                          # camera 0, all classes
./build/tracker --classes person,cup     # only these classes
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

## Custom labels
Teach it a specific item, such as your mug or a person's face, from a few photos:
```sh
./build/tracker --enroll my_mug mug1.jpg mug2.jpg mug3.jpg mug4.jpg mug5.jpg
./build/tracker --enroll alex alex1.jpg alex2.jpg alex3.jpg       # a face: it finds the face itself
./build/tracker --list-labels
./build/tracker --classes my_mug          # detect cups, show only the one that is my_mug
./build/tracker --remove-label my_mug     # deletes the label and its copied photos
```
- It finds the item in each photo with the normal detectors (add `--class cup` to say what it is). It copies the photos to `labels/<name>/` and stores a neural-network *embedding* of each, plus a mirrored copy, in `label.yml`.
- Embeddings come from SFace for faces and DINOv2 for objects. Enrolling the same name again adds photos.
- Photos where the item can't be found are skipped. For close-ups that show only the item, use `--whole-photo`.
- **5–10 photos from different angles and in different light work best.** Enrollment reports how well the photos agree with each other and points out any that don't fit.
- While tracking, objects of a labeled class are checked in the background a few times a second. A track takes the label after 3 matching checks, and each label is one item, so at most one track has it.
- Labeled tracks show the label instead of `class #n`. A face under a bright light still shows `TARGET`.
- `labels/` holds your photos and isn't committed to git.

Telling *a* mug from *your* mug works when the photos cover the angles you'll show it from. Two identical products (two of the same phone) can't be told apart from pictures alone.

## Logs
Each run writes `logs/run_<date>_<time>.csv`: one row per visible track per frame. The columns are `time_ms`, `frame`, `id` (unique for the run), `number` (the one on screen), `class`, `label`, `matched` (0 = predicted box), the box, `conf`, `target` and `light_ratio` (faces). To summarize one or more runs:
```sh
scripts/log_stats.py logs/run_*.csv
```
It prints the frame rate, the number of IDs per class and how long tracks lasted. Fewer IDs and longer lifetimes mean steadier tracking.

To tune the tracker without re-running the neural networks, save a clip's detections once and replay them:
```sh
./build/tracker --video clips/hands_face.mov --no-window --no-log --dump-detections /tmp/dets.csv
./build/tracker_replay /tmp/dets.csv --tracks lostTimeout=1.5   # any TrackerParams field
```

## Layout
```
src/        Camera (capture thread); ClassRegistry (all class ids); Detector runs the IDetector
            implementations (YoloDetector, FaceDetector, HandDetector) side by side; OrtModel
            (ONNX Runtime + CoreML); Tracker (+ KalmanBox, Assignment); Targeting (TARGET);
            Embedder, Labels, Enrollment, Recognizer (custom labels); Options; Overlay
            (+ TagLayout); Logger; main
labels/     custom labels: photos + label.yml per label (not in git)
tools/      replay.cpp: the tracker on saved detections (tracker_replay)
tests/      GoogleTest unit tests (tracker_tests)
scripts/    fetch_models.sh, export_yolo.sh, log_stats.py
models/     coco.names (tracked); *.onnx (downloaded, not in git)
clips/      reference videos for regression runs (not in git)
logs/       run logs (not in git)
```
