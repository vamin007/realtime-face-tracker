"""Draws synthetic laser dots onto a clip, placed from the tracker's own face
detections (a --dump-detections CSV), and writes the result plus a schedule.

usage: make_laser_clip.py VIDEO DETS_CSV OUT_VIDEO OUT_SCHEDULE_CSV
"""
import csv
import sys

import cv2
import numpy as np

video, dets_path, out_path, schedule_path = sys.argv[1:5]

# (start s, end s, where): where the dot goes in that segment.
SEGMENTS = [
    (2, 5, "cheek"), (7, 10, "wall_in_box"), (12, 15, "above_hairline"),
    (17, 20, "forehead"), (22, 25, "beside_head"), (27, 30, "nose"),
    (32, 35, "box_bottom_corner"), (37, 40, "cheek"),
]
ON_FACE = {"cheek", "forehead", "nose"}

faces = {}  # frame -> (box, keypoints) of its most confident face
for row in csv.DictReader(open(dets_path)):
    if row["class"] != "face" or row["weak"] == "1":
        continue
    f = int(row["frame"])
    conf = float(row["conf"])
    if f in faces and faces[f][2] >= conf:
        continue
    box = [float(row[k]) for k in ("x", "y", "w", "h")]
    kp = np.array([float(v) for v in row["keypoints"].split()]).reshape(-1, 2)
    faces[f] = (box, kp, conf)


def spot(where, box, kp):
    x, y, w, h = box
    eye_mid = (kp[0] + kp[1]) / 2
    mouth_mid = (kp[3] + kp[4]) / 2
    eye_dist = np.linalg.norm(kp[1] - kp[0])
    m = np.linalg.norm(mouth_mid - eye_mid)
    up = (eye_mid - mouth_mid) / m
    if where == "cheek":
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
    if where == "wall_in_box":
        return np.array([x + 0.06 * w, y + 0.06 * h])
    if where == "box_bottom_corner":
        return np.array([x + 0.94 * w, y + 0.94 * h])
    raise ValueError(where)


def draw_laser(frame, p):
    # A red glow with an overexposed middle, slightly bloomed like a camera sees it.
    c = (int(round(p[0])), int(round(p[1])))
    x0, y0 = max(c[0] - 12, 0), max(c[1] - 12, 0)
    layer = frame.copy()
    cv2.circle(layer, c, 5, (40, 40, 240), -1, cv2.LINE_AA)
    cv2.circle(layer, c, 2, (245, 245, 255), -1, cv2.LINE_AA)
    roi = layer[y0:c[1] + 13, x0:c[0] + 13]
    layer[y0:c[1] + 13, x0:c[0] + 13] = cv2.GaussianBlur(roi, (3, 3), 0)
    frame[:] = layer


cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
with open(schedule_path, "w") as sched:
    sched.write("frame,where,on_face,x,y\n")
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = f / fps
        where = next((s[2] for s in SEGMENTS if s[0] <= t < s[1]), None)
        if where and f in faces:
            box, kp, _ = faces[f]
            p = spot(where, box, kp)
            draw_laser(frame, p)
            sched.write(f"{f},{where},{int(where in ON_FACE)},{p[0]:.1f},{p[1]:.1f}\n")
        out.write(frame)
        f += 1
out.release()
print(f"wrote {f} frames to {out_path}")
