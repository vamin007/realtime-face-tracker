#!/usr/bin/env bash
# Downloads the models the tracker uses into models/ and checks each SHA-256.
# Files that are already present with the right checksum are skipped.
# YOLO is not downloaded here: it is exported locally by scripts/export_yolo.sh.
# The face detector (YuNet) and face recognizer (SFace) come from the OpenCV model zoo; the
# tracker looks for them in models/ (see Options.cpp for the default file names).
# -e: stop at the first failing command; -u: unset variables are errors; pipefail: a failure
# inside a pipeline counts.
set -euo pipefail

# Work in models/ whatever directory the script was started from.
cd "$(dirname "$0")/../models"

# Sources are pinned to a commit so a later upstream change can't swap a file.
ZOO=https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models

# What to fetch, one entry per model: file | url | sha256 (the expected checksum).
MODELS=(
    "face_detection_yunet_2023mar.onnx|$ZOO/face_detection_yunet/face_detection_yunet_2023mar.onnx|8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    "face_recognition_sface_2021dec.onnx|$ZOO/face_recognition_sface/face_recognition_sface_2021dec.onnx|0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
)

# Prints the SHA-256 checksum of a file (shasum ships with macOS).
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

# For each model: skip it if it is already here and correct; otherwise download and verify it.
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name url want <<< "$entry"
    if [[ -f "$name" && "$(sha "$name")" == "$want" ]]; then
        echo "ok        $name"
        continue
    fi
    echo "fetching  $name"
    # Download to a .part file first, so an interrupted download never leaves a truncated model
    # under the real name. -f: fail on HTTP errors; -L: follow redirects.
    curl -fL --progress-bar -o "$name.part" "$url"
    # Verify before the file is moved into place; a wrong checksum means a corrupted or
    # tampered download.
    got=$(sha "$name.part")
    if [[ "$got" != "$want" ]]; then
        rm -f "$name.part"
        echo "error: checksum mismatch for $name (got $got)" >&2
        exit 1
    fi
    mv "$name.part" "$name"
    echo "ok        $name"
done

# YOLO has no download: this just reports whether it has been exported yet (the tracker falls
# back to this square 640x640 model when the 16:9 one is missing). The 384x640 export is
# separate: scripts/export_yolo.sh 384 640.
if [[ -f yolov8n.onnx ]]; then
    echo "ok        yolov8n.onnx (exported locally, not checksummed)"
else
    echo "missing   yolov8n.onnx: run scripts/export_yolo.sh to create it" >&2
    exit 1
fi
