#!/usr/bin/env bash
# Exports YOLOv8n to ONNX with Ultralytics, using a project-local virtualenv (.venv).
# Usage: scripts/export_yolo.sh [HEIGHT WIDTH]     default 640 640
#   640 640 -> models/yolov8n.onnx
#   384 640 -> models/yolov8n_384x640.onnx   (16:9-friendly input, used from M2)
# The first run installs Ultralytics and PyTorch into .venv (a large download).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
H=${1:-640}
W=${2:-640}
if [[ "$H" == 640 && "$W" == 640 ]]; then
    OUT="$ROOT/models/yolov8n.onnx"
else
    OUT="$ROOT/models/yolov8n_${H}x${W}.onnx"
fi
if [[ -e "$OUT" ]]; then
    echo "error: $OUT already exists; move it away first" >&2
    exit 1
fi

VENV="$ROOT/.venv"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet ultralytics onnx
"$VENV/bin/python" -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

# Export in a temp dir: Ultralytics downloads yolov8n.pt next to where it runs.
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
"$VENV/bin/yolo" export model=yolov8n.pt format=onnx imgsz="$H,$W"
mv yolov8n.onnx "$OUT"
echo "wrote $OUT"
