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

Why: without real laser footage this builds test clips for the targeting. A dot's position is
computed each frame from that frame's face landmarks, so it follows the face; PLANS says where
on or near the face it goes during each time segment. Dots on the face (cheek, forehead, nose)
should make it the TARGET, the other spots (wall, above the hairline, beside the head, the
shirt, the red mark) should not. Needs OpenCV and NumPy; the result is meant for
`tracker --video OUT_VIDEO.mkv --dump-laser ...`.
"""
import argparse
import csv
import sys

import cv2
import numpy as np

# (start s, end s, where): where the dot goes in that segment. The names are positions
# relative to the face landmarks (see spot()); outside every segment there is no dot.
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
# Positions that are on the face itself: the face should become the TARGET there. Everything
# else, including red_mark (pigment, not light), should not.
ON_FACE = {"cheek", "forehead", "nose"}

# The laser, in linear light (1 = full scale): peak, width (px) and how much of
# it reaches each channel (B, G, R); and its bloom.
LASER_PEAK, LASER_SIGMA = 25.0, 1.3
CROSSTALK = np.array([0.05, 0.08, 1.0])
BLOOM_PEAK, BLOOM_SIGMA = 0.12, 5.0
# The red mark: radius (px) and how much light it reflects next to the skin (B, G, R).
MARK_RADIUS = 4.5
MARK_REFLECTANCE = np.array([0.35, 0.30, 1.25])

# Pixel values are gamma-encoded (value ~ light ** (1 / 2.2)), but light adds up in linear
# space: dots are added there. TO_LINEAR converts 8-bit values to linear light for that, and
# to_frame converts back.
TO_LINEAR = (np.arange(256) / 255.0) ** 2.2


def to_frame(linear):
    return np.round(255 * np.clip(linear, 0, 1) ** (1 / 2.2)).astype(np.uint8)


# A square window of radius r around p, clipped to the frame: the slices that index it and
# each pixel's squared distance from p.
def patch_around(frame, p, r):
    h, w = frame.shape[:2]
    x0, x1 = max(int(p[0]) - r, 0), min(int(p[0]) + r + 1, w)
    y0, y1 = max(int(p[1]) - r, 0), min(int(p[1]) + r + 1, h)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return (slice(y0, y1), slice(x0, x1)), (xs - p[0]) ** 2 + (ys - p[1]) ** 2


# Adds a laser spot at p to the frame in place: a narrow Gaussian core (bright enough to saturate
# red, and, through CROSSTALK, to leak into green and blue) plus a faint wide bloom. strength
# scales it.
def add_laser(frame, p, strength):
    where, d2 = patch_around(frame, p, 20)
    light = (strength * LASER_PEAK * np.exp(-d2 / (2 * LASER_SIGMA ** 2)) +
             BLOOM_PEAK * np.exp(-d2 / (2 * BLOOM_SIGMA ** 2)))
    frame[where] = to_frame(TO_LINEAR[frame[where]] + light[..., None] * CROSSTALK)


# Paints a red mark (pigment) at p in place: inside a disc with a soft edge it multiplies the
# skin's light by MARK_REFLECTANCE, so it stays lit like the skin around it and doesn't glow.
def add_mark(frame, p):
    where, d2 = patch_around(frame, p, int(MARK_RADIUS) + 2)
    cover = np.clip(MARK_RADIUS + 0.5 - np.sqrt(d2), 0, 1)[..., None]  # soft edge
    frame[where] = to_frame(TO_LINEAR[frame[where]] * (1 + cover * (MARK_REFLECTANCE - 1)))


def camera_colors(frame, p, r=24):
    """Subsamples the colors around p like a camera's 4:2:0 feed.

    Video stores brightness at full resolution but the two color channels (U, V) at half
    resolution in each direction (chroma subsampling), so a dot only a pixel or two wide has
    its color smeared over 2x2 blocks. Converting to YUV I420 and back reproduces that. The
    window starts and ends on even pixels because I420 needs even sizes.
    """
    h, w = frame.shape[:2]
    x0, y0 = max(int(p[0]) - r, 0) & ~1, max(int(p[1]) - r, 0) & ~1
    x1, y1 = min(x0 + 2 * r, w) & ~1, min(y0 + 2 * r, h) & ~1
    yuv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2YUV_I420)
    frame[y0:y1, x0:x1] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


def skin_red(frame, p):
    """Median red of the ring 8-14 px around p.

    Measured before the dot is drawn, so it is the skin's own red (channel 2 in BGR); it goes
    into the schedule to compare results with how bright the skin was.
    """
    where, d2 = patch_around(frame, p, 14)
    ring = (d2 >= 64) & (d2 <= 196)
    return float(np.median(frame[where][..., 2][ring])) if ring.any() else float("nan")


# Where `where` puts the dot, in frame pixels. box is the face detection (x, y, w, h); kp are
# its 5 landmarks (right eye, left eye, nose, right and left mouth corner). Positions are
# built from the landmarks, so they scale with the face (eye_dist, eye-to-mouth distance m)
# and follow its tilt (`up`, the unit vector from the mouth toward the eyes).
def spot(where, box, kp):
    x, y, w, h = box
    eye_mid = (kp[0] + kp[1]) / 2
    mouth_mid = (kp[3] + kp[4]) / 2
    eye_dist = np.linalg.norm(kp[1] - kp[0])
    m = np.linalg.norm(mouth_mid - eye_mid)
    up = (eye_mid - mouth_mid) / m
    # Cheek: halfway between an eye and the mouth corner below it, moved a bit out from the nose.
    if where in ("cheek", "red_mark"):
        mid = (kp[1] + kp[4]) / 2
        side = np.sign(mid[0] - kp[2][0]) or 1
        return mid + np.array([side * 0.25 * eye_dist, 0])
    # Forehead: half an eyes-to-mouth distance above the eyes.
    if where == "forehead":
        return eye_mid + up * 0.5 * m
    # Nose: just above its tip.
    if where == "nose":
        return kp[2] + up * 0.15 * m
    # Above the hairline: hair or wall, off the face.
    if where == "above_hairline":
        return eye_mid + up * 1.3 * m
    # Beside the head: halfway between eyes and mouth in height, 1.3 eye distances to the right.
    if where == "beside_head":
        return (eye_mid + mouth_mid) / 2 + np.array([1.3 * eye_dist, 0])
    # On the shirt, far below the chin.
    if where == "shirt":
        return mouth_mid - up * 2.6 * m
    # The box's top-left corner: wall inside the detection box but off the face.
    if where == "wall_in_box":
        return np.array([x + 0.06 * w, y + 0.06 * h])
    # The box's bottom-right corner: also inside the box and off the face.
    if where == "box_bottom_corner":
        return np.array([x + 0.94 * w, y + 0.94 * h])
    raise ValueError(where)


# Command line: the clip, the tracker's detections CSV, the output video and the schedule.
parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("video")
parser.add_argument("dets")
parser.add_argument("out")
parser.add_argument("schedule")
parser.add_argument("--plan", choices=sorted(PLANS), default="default")
args = parser.parse_args()
segments = PLANS[args.plan]

# Load the tracker's face detections: per frame, keep only the most confident face (weak
# detections are skipped).
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

# A fixed seed, so the same inputs always make the same clip (the dot strength flickers).
rng = np.random.default_rng(1)
# Read the clip frame by frame and write it back, losslessly, with the dots drawn on.
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
        # The plan segment this frame's time falls in, if any.
        where = next((s[2] for s in segments if s[0] <= t < s[1]), None)
        # A dot needs a segment and a face detected in this frame (its landmarks give the spot).
        if where and f in faces:
            box, kp, _ = faces[f]
            p = spot(where, box, kp)
            if 0 <= p[0] < w and 0 <= p[1] < h:
                # Measure the skin before drawing on it.
                skin = skin_red(frame, p)
                if where == "red_mark":
                    add_mark(frame, p)
                else:
                    add_laser(frame, p, rng.uniform(0.6, 1.4))  # a hand-held laser flickers
                # Last, degrade the colors around it the way a camera feed does.
                camera_colors(frame, p)
                # Schedule row: frame, position, on_face, x, y, skin red.
                sched.write(f"{f},{where},{int(where in ON_FACE)},{p[0]:.1f},{p[1]:.1f},"
                            f"{skin:.0f}\n")
        out.write(frame)
        f += 1
out.release()
print(f"wrote {f} frames to {args.out}")
