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
./build/tracker --conf-hand 0.5                           # per-detector thresholds: --conf-yolo/-face/-hand
./build/tracker --list-classes
./build/tracker --help                   # all options
```
Keys: `q` or `Esc` quits.
The first camera run asks for camera access: allow it for your terminal app in **System Settings → Privacy & Security → Camera**.

At startup it prints where each detector runs. CoreML compiles a model on first use (a few seconds) and
keeps the result in `.cache/coreml/`. A model runs on CoreML only if CoreML can take all of it; otherwise
it runs on the CPU, which was faster for the partly supported palm model.

## Logs
Each run writes `logs/run_<date>_<time>.csv`. To summarize one or more runs:
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
            Overlay; Logger; main
tools/      replay.cpp: the tracker on saved detections (tracker_replay)
tests/      GoogleTest unit tests (tracker_tests)
scripts/    fetch_models.sh, export_yolo.sh, log_stats.py
models/     coco.names (tracked); *.onnx (downloaded, not in git)
clips/      reference videos for regression runs (not in git)
logs/       run logs (not in git)
```
