#!/usr/bin/env bash
# Exports YOLOv8n to ONNX with Ultralytics, using a project-local virtualenv (.venv).
# Usage: scripts/export_yolo.sh [HEIGHT WIDTH]     default 640 640
#   640 640 -> models/yolov8n.onnx
#   384 640 -> models/yolov8n_384x640.onnx   (16:9-friendly input, used from M2)
# The first run installs Ultralytics and PyTorch into .venv (a large download).
# Why export it here: Ultralytics distributes PyTorch weights (yolov8n.pt); the tracker runs
# the ONNX format through ONNX Runtime. The export fixes the input size, and the tracker reads
# it back from the model (see YoloDetector), so a different size needs no code change.
# -e: stop at the first failing command; -u: unset variables are errors; pipefail: a failure
# inside a pipeline counts.
set -euo pipefail

# Project root (this script lives in scripts/), the requested input size, and the output name:
# the default square size keeps the plain name, any other size is added to it.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
H=${1:-640}
W=${2:-640}
if [[ "$H" == 640 && "$W" == 640 ]]; then
    OUT="$ROOT/models/yolov8n.onnx"
else
    OUT="$ROOT/models/yolov8n_${H}x${W}.onnx"
fi
# Never overwrite an existing model: the caller has to move the old file away on purpose.
if [[ -e "$OUT" ]]; then
    echo "error: $OUT already exists; move it away first" >&2
    exit 1
fi

# A project-local virtualenv keeps the large export tooling out of the system Python.
VENV="$ROOT/.venv"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet ultralytics onnx
"$VENV/bin/python" -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

# Export in a temp dir: Ultralytics downloads yolov8n.pt next to where it runs. The trap
# deletes the temp dir when the script ends, even on an error.
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
# imgsz is "height,width". The output is a [1, 84, N] tensor: 4 box values + 80 COCO class
# scores for each of N candidate boxes (see decodeYolo in YoloDetector.cpp).
"$VENV/bin/yolo" export model=yolov8n.pt format=onnx imgsz="$H,$W"
mv yolov8n.onnx "$OUT"
echo "wrote $OUT"
