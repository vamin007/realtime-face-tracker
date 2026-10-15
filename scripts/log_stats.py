#!/usr/bin/env python3
"""Summarize tracker CSV logs: frame rate, track IDs per class, and track lifetimes.

Usage: scripts/log_stats.py logs/run_*.csv

Fewer IDs per class and longer lifetimes mean steadier tracking. Short tracks
(5 frames or fewer) are mostly identity breaks or noise.

Reads the original columns (timestamp_ms,frame,id,class,...) and the v2 columns
(t_capture_ms,frame,uid,type,...).
"""
import csv
import statistics
import sys
from collections import defaultdict

SHORT_FRAMES = 5


def first_column(row, *names):
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"none of {names} in log columns {list(row)}")


def summarize(path):
    frames = set()
    times = []
    span = {}  # (class, id) -> [first frame, last frame]
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            frames.add(frame)
            times.append(int(first_column(row, "t_capture_ms", "timestamp_ms")))
            key = (first_column(row, "type", "class"), first_column(row, "uid", "id"))
            if key in span:
                span[key][1] = frame
            else:
                span[key] = [frame, frame]

    print(f"== {path}")
    if not frames:
        print("   (no tracks logged)")
        return
    seconds = (max(times) - min(times)) / 1000
    nframes = max(frames) - min(frames) + 1
    fps = nframes / seconds if seconds > 0 else float("nan")
    print(f"   frames {nframes}, {seconds:.1f} s, ~{fps:.1f} FPS (frames with no tracks still count)")

    by_class = defaultdict(list)
    for (cls, _), (first, last) in span.items():
        by_class[cls].append(last - first + 1)

    print(f"   {'class':<14}{'IDs':>5}{'short':>7}{'median life':>13}{'longest':>9}")
    for cls in sorted(by_class, key=lambda c: -len(by_class[c])):
        lives = by_class[cls]
        short = sum(1 for n in lives if n <= SHORT_FRAMES)
        print(f"   {cls:<14}{len(lives):>5}{short:>7}{statistics.median(lives):>13.0f}{max(lives):>9}")
    lives = [n for v in by_class.values() for n in v]
    short = sum(1 for n in lives if n <= SHORT_FRAMES)
    print(f"   {'TOTAL':<14}{len(lives):>5}{short:>7}{statistics.median(lives):>13.0f}{max(lives):>9}")


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 1
    for path in sys.argv[1:]:
        summarize(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
