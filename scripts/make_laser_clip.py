"""Draws synthetic laser dots onto a clip, placed from the tracker's own face
detections (a --dump-detections CSV), and writes the result plus a schedule.

A laser adds red light. Each dot is added in linear light: a narrow Gaussian
spot, bright enough to saturate the red channel (and, through the sensor's
crosstalk, green and blue in the middle: a white core), plus a faint wide
bloom. Green and blue rise much less than red, and on bright skin red can only
rise to 255, so a dot there is less "redder than the skin" than on dim skin.
The `red_mark` segment is a static red mark instead (pigment: a red sticker, a
bindi): it changes how much of the skin's light is reflected, so it is lit like
the skin around it and doesn't glow.

Around each dot the colors are subsampled like a camera's 4:2:0 video feed, and
the result is written losslessly (FFV1; use a .mkv name): a lossy re-encode
would wash out a small dot's red rim far more than a camera does.

usage: make_laser_clip.py VIDEO DETS_CSV OUT_VIDEO.mkv OUT_SCHEDULE_CSV [--plan flashlight]

The schedule has one row per frame with a dot or mark: where it is, whether it
is on the face (should become TARGET), and the skin's median red around it.
"""
import argparse
import csv
import sys

import cv2
import numpy as np

# (start s, end s, where): where the dot goes in that segment.
PLANS = {
    "default": [
        (2, 5, "cheek"), (7, 10, "wall_in_box"), (12, 15, "above_hairline"),
        (17, 20, "forehead"), (22, 25, "beside_head"), (27, 30, "nose"),
        (32, 35, "box_bottom_corner"), (37, 40, "cheek"), (42, 45, "red_mark"),
        (47, 50, "shirt"),
    ],
    # flashlight.mov: the on-face dots and the red mark fall where the
    # flashlight lights the face (skin red about 200-220).
    "flashlight": [
        (5.6, 7.1, "cheek"), (7.6, 9.6, "forehead"), (11.0, 12.3, "nose"),
        (13, 16, "wall_in_box"), (18.4, 20.0, "cheek"), (21, 24, "above_hairline"),
        (25.5, 28.8, "red_mark"), (30, 33, "beside_head"), (34, 37, "box_bottom_corner"),
        (38, 41, "shirt"), (46.9, 48.0, "forehead"),
    ],
}
ON_FACE = {"cheek", "forehead", "nose"}

# The laser, in linear light (1 = full scale): peak, width (px) and how much of
# it reaches each channel (B, G, R); and its bloom.
LASER_PEAK, LASER_SIGMA = 25.0, 1.3
CROSSTALK = np.array([0.05, 0.08, 1.0])
BLOOM_PEAK, BLOOM_SIGMA = 0.12, 5.0
# The red mark: radius (px) and how much light it reflects next to the skin (B, G, R).
MARK_RADIUS = 4.5
MARK_REFLECTANCE = np.array([0.35, 0.30, 1.25])

TO_LINEAR = (np.arange(256) / 255.0) ** 2.2


def to_frame(linear):
    return np.round(255 * np.clip(linear, 0, 1) ** (1 / 2.2)).astype(np.uint8)


def patch_around(frame, p, r):
    h, w = frame.shape[:2]
    x0, x1 = max(int(p[0]) - r, 0), min(int(p[0]) + r + 1, w)
    y0, y1 = max(int(p[1]) - r, 0), min(int(p[1]) + r + 1, h)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return (slice(y0, y1), slice(x0, x1)), (xs - p[0]) ** 2 + (ys - p[1]) ** 2


def add_laser(frame, p, strength):
    where, d2 = patch_around(frame, p, 20)
    light = (strength * LASER_PEAK * np.exp(-d2 / (2 * LASER_SIGMA ** 2)) +
             BLOOM_PEAK * np.exp(-d2 / (2 * BLOOM_SIGMA ** 2)))
    frame[where] = to_frame(TO_LINEAR[frame[where]] + light[..., None] * CROSSTALK)


def add_mark(frame, p):
    where, d2 = patch_around(frame, p, int(MARK_RADIUS) + 2)
    cover = np.clip(MARK_RADIUS + 0.5 - np.sqrt(d2), 0, 1)[..., None]  # soft edge
    frame[where] = to_frame(TO_LINEAR[frame[where]] * (1 + cover * (MARK_REFLECTANCE - 1)))


def camera_colors(frame, p, r=24):
    """Subsamples the colors around p like a camera's 4:2:0 feed."""
    h, w = frame.shape[:2]
    x0, y0 = max(int(p[0]) - r, 0) & ~1, max(int(p[1]) - r, 0) & ~1
    x1, y1 = min(x0 + 2 * r, w) & ~1, min(y0 + 2 * r, h) & ~1
    yuv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2YUV_I420)
    frame[y0:y1, x0:x1] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


def skin_red(frame, p):
    """Median red of the ring 8-14 px around p."""
    where, d2 = patch_around(frame, p, 14)
    ring = (d2 >= 64) & (d2 <= 196)
    return float(np.median(frame[where][..., 2][ring])) if ring.any() else float("nan")


def spot(where, box, kp):
    x, y, w, h = box
    eye_mid = (kp[0] + kp[1]) / 2
    mouth_mid = (kp[3] + kp[4]) / 2
    eye_dist = np.linalg.norm(kp[1] - kp[0])
    m = np.linalg.norm(mouth_mid - eye_mid)
    up = (eye_mid - mouth_mid) / m
    if where in ("cheek", "red_mark"):
        mid = (kp[1] + kp[4]) / 2
        side = np.sign(mid[0] - kp[2][0]) or 1
        return mid + np.array([side * 0.25 * eye_dist, 0])
    if where == "forehead":
        return eye_mid + up * 0.5 * m
    if where == "nose":
        return kp[2] + up * 0.15 * m
    if where == "above_hairline":
        return eye_mid + up * 1.3 * m
    if where == "beside_head":
        return (eye_mid + mouth_mid) / 2 + np.array([1.3 * eye_dist, 0])
    if where == "shirt":
        return mouth_mid - up * 2.6 * m
    if where == "wall_in_box":
        return np.array([x + 0.06 * w, y + 0.06 * h])
    if where == "box_bottom_corner":
        return np.array([x + 0.94 * w, y + 0.94 * h])
    raise ValueError(where)


parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("video")
parser.add_argument("dets")
parser.add_argument("out")
parser.add_argument("schedule")
parser.add_argument("--plan", choices=sorted(PLANS), default="default")
args = parser.parse_args()
segments = PLANS[args.plan]

faces = {}  # frame -> (box, keypoints) of its most confident face
for row in csv.DictReader(open(args.dets)):
    if row["class"] != "face" or row["weak"] == "1":
        continue
    f = int(row["frame"])
    conf = float(row["conf"])
    if f in faces and faces[f][2] >= conf:
        continue
    box = [float(row[k]) for k in ("x", "y", "w", "h")]
    kp = np.array([float(v) for v in row["keypoints"].split()]).reshape(-1, 2)
    faces[f] = (box, kp, conf)

rng = np.random.default_rng(1)
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(args.out, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
if not out.isOpened():
    sys.exit(f"can't write {args.out}")
with open(args.schedule, "w") as sched:
    sched.write("frame,where,on_face,x,y,skin_r\n")
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = f / fps
        where = next((s[2] for s in segments if s[0] <= t < s[1]), None)
        if where and f in faces:
            box, kp, _ = faces[f]
            p = spot(where, box, kp)
            if 0 <= p[0] < w and 0 <= p[1] < h:
                skin = skin_red(frame, p)
                if where == "red_mark":
                    add_mark(frame, p)
                else:
                    add_laser(frame, p, rng.uniform(0.6, 1.4))  # a hand-held laser flickers
                camera_colors(frame, p)
                sched.write(f"{f},{where},{int(where in ON_FACE)},{p[0]:.1f},{p[1]:.1f},"
                            f"{skin:.0f}\n")
        out.write(frame)
        f += 1
out.release()
print(f"wrote {f} frames to {args.out}")
