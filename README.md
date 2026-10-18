# Object Tracker

Tracks objects through the MacBook camera (or a video file) and draws a live overlay with an ID per object.
Each run also writes a CSV log of every visible track.

- **Detection:** YOLOv8n for the 80 COCO classes, YuNet for faces, MediaPipe palm detection for hands.
  The three run side by side; YOLO runs on the GPU / Neural Engine through ONNX Runtime + CoreML.
- **Tracking:** SORT-style: one Kalman filter per track, matched to new detections by box overlap (IoU).

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

## Layout
```
src/        Camera (capture thread); ClassRegistry (all class ids); Detector runs the IDetector
            implementations (YoloDetector, FaceDetector, HandDetector) side by side; OrtModel
            (ONNX Runtime + CoreML); Tracker; Overlay; Logger; main
tests/      GoogleTest unit tests (tracker_tests)
scripts/    fetch_models.sh, export_yolo.sh, log_stats.py
models/     coco.names (tracked); *.onnx (downloaded, not in git)
clips/      reference videos for regression runs (not in git)
logs/       run logs (not in git)
```
