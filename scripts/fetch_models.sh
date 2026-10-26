#!/usr/bin/env bash
# Downloads the models the tracker uses into models/ and checks each SHA-256.
# Files that are already present with the right checksum are skipped.
# YOLO is not downloaded here: it is exported locally by scripts/export_yolo.sh.
set -euo pipefail

cd "$(dirname "$0")/../models"

# Sources are pinned to a commit so a later upstream change can't swap a file.
ZOO=https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models

# file | url | sha256
MODELS=(
    "face_detection_yunet_2023mar.onnx|$ZOO/face_detection_yunet/face_detection_yunet_2023mar.onnx|8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    "palm_detection_mediapipe_2023feb.onnx|$ZOO/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx|78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c"
    "face_recognition_sface_2021dec.onnx|$ZOO/face_recognition_sface/face_recognition_sface_2021dec.onnx|0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
)

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name url want <<< "$entry"
    if [[ -f "$name" && "$(sha "$name")" == "$want" ]]; then
        echo "ok        $name"
        continue
    fi
    echo "fetching  $name"
    curl -fL --progress-bar -o "$name.part" "$url"
    got=$(sha "$name.part")
    if [[ "$got" != "$want" ]]; then
        rm -f "$name.part"
        echo "error: checksum mismatch for $name (got $got)" >&2
        exit 1
    fi
    mv "$name.part" "$name"
    echo "ok        $name"
done

if [[ -f yolov8n.onnx ]]; then
    echo "ok        yolov8n.onnx (exported locally, not checksummed)"
else
    echo "missing   yolov8n.onnx: run scripts/export_yolo.sh to create it" >&2
    exit 1
fi
