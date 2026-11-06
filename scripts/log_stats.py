#!/usr/bin/env python3
"""Summarize tracker CSV logs: frame rate, track IDs per class, track lifetimes and flicker.

Usage: scripts/log_stats.py logs/run_*.csv

Fewer IDs per class and longer lifetimes mean steadier tracking. Short tracks
(5 frames or fewer) are mostly identity breaks or noise.

A log is the CSV every tracker run writes to logs/ (one row per visible track per frame).
The output is plain text, one block per file. tools/replay.cpp measures the same flicker
from saved detections instead of from a log.

Flicker, per class (newer logs only; all counts are per minute of the clip):
  gaps      a track was hidden and came back (visible -> hidden -> visible)
  thin      a shown box switched to a thin, predicted box (coasting 0 -> 1; logs
            without that column: matched 1 -> 0)
  thin%     share of shown boxes that were thin
  jit50/90  box wobble: the change in the box center's motion from one frame to
            the next (second difference), in % of the box size; median and 90th
            percentile. Steady motion scores 0, so this is noise, not movement.
  size50/90 frame-to-frame change of width and height, in %
  cls/lbl/num  changes of a track's class, custom label or number

Reads the original columns (timestamp_ms,frame,id,class,...) and the newer ones
(time_ms,frame,id,number,class,...). Original logs stamp each row with the wall
clock, so their FPS is the processing speed; newer logs use each frame's own time
(a video's position), so for a video the FPS is the video's frame rate.
"""
import csv
import math
import statistics
import sys
from collections import defaultdict

# A track shown for this many frames or fewer counts as "short".
SHORT_FRAMES = 5


# Returns row[name] for the first of `names` the log has: older and newer logs name some
# columns differently (timestamp_ms / time_ms).
def first_column(row, *names):
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"none of {names} in log columns {list(row)}")


# The p-th percentile (0-100) of values by rank, without interpolation; NaN when empty.
def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def flicker(rows_by_track, minutes):
    """Flicker counts per class from each track's rows (frame order)."""
    # by_class[class][measure] is a list: one count per track (gaps, thin, cls, lbl, num) or one
    # value per frame (jitter, size). They are summed or taken as percentiles when printing.
    by_class = defaultdict(lambda: defaultdict(list))
    for rows in rows_by_track.values():
        # A track counts under the class it ended with (a class change is what `cls` counts).
        stats = by_class[rows[-1]["class"]]
        gaps = thin = cls = lbl = num = 0
        for prev, row in zip(rows, rows[1:]):
            # A skipped frame number means the track was hidden and then came back.
            gaps += row["frame"] > prev["frame"] + 1
            # thin counts only between consecutive frames (after a gap there is no "switch").
            if row["frame"] == prev["frame"] + 1:
                thin += row["thin"] and not prev["thin"]
            cls += row["class"] != prev["class"]
            lbl += row["label"] != prev["label"]
            num += row["number"] != prev["number"]
        thin_rows = sum(r["thin"] for r in rows)
        for key, value in (("gaps", gaps), ("thin", thin), ("thin_rows", thin_rows),
                           ("rows", len(rows)), ("cls", cls), ("lbl", lbl), ("num", num)):
            stats[key].append(value)
        # Wobble: the second difference of the box center over three consecutive frames, c - 2b + a,
        # in % of the box size. It is 0 for motion at constant velocity, so it measures noise, not
        # movement.
        for a, b, c in zip(rows, rows[1:], rows[2:]):
            if c["frame"] != a["frame"] + 2:
                continue
            size = math.sqrt(max(b["w"] * b["h"], 1))
            jx = c["cx"] - 2 * b["cx"] + a["cx"]
            jy = c["cy"] - 2 * b["cy"] + a["cy"]
            stats["jitter"].append(100 * math.hypot(jx, jy) / size)
        # Frame-to-frame change of width and height, in % of the earlier value.
        for a, b in zip(rows, rows[1:]):
            if b["frame"] != a["frame"] + 1:
                continue
            for dim in ("w", "h"):
                stats["size"].append(100 * abs(b[dim] - a[dim]) / max(a[dim], 1))

    # One row per class (most rows first), then a TOTAL over all classes.
    print(f"   {'flicker /min':<14}{'gaps':>6}{'thin':>6}{'thin%':>7}{'jit50':>7}{'jit90':>7}"
          f"{'size50':>8}{'size90':>8}{'cls':>5}{'lbl':>5}{'num':>5}")
    totals = defaultdict(list)
    for cls in sorted(by_class, key=lambda c: -sum(by_class[c]["rows"])):
        stats = by_class[cls]
        for key, values in stats.items():
            totals[key].extend(values)
        print_flicker_row(cls, stats, minutes)
    print_flicker_row("TOTAL", totals, minutes)


# One row of the flicker table: events per minute, the share of rows drawn thin, and the
# percentiles of wobble (jit) and size change.
def print_flicker_row(name, stats, minutes):
    per_min = lambda key: sum(stats[key]) / minutes
    thin_share = 100 * sum(stats["thin_rows"]) / max(sum(stats["rows"]), 1)
    print(f"   {name:<14}{per_min('gaps'):>6.1f}{per_min('thin'):>6.1f}{thin_share:>6.1f}%"
          f"{percentile(stats['jitter'], 50):>7.1f}{percentile(stats['jitter'], 90):>7.1f}"
          f"{percentile(stats['size'], 50):>8.1f}{percentile(stats['size'], 90):>8.1f}"
          f"{per_min('cls'):>5.1f}{per_min('lbl'):>5.1f}{per_min('num'):>5.1f}")


# Prints the summary of one log file.
def summarize(path):
    frames = set()
    times = []
    span = {}  # (class, id) -> [first frame, last frame]
    rows_by_track = defaultdict(list)  # id -> rows, newer logs only
    # One pass over the rows: frame numbers and times, each track's first and last frame, and
    # (newer logs only) the per-frame values that flicker() needs.
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            frames.add(frame)
            times.append(int(first_column(row, "time_ms", "timestamp_ms")))
            key = (row["class"], row["id"])
            if key in span:
                span[key][1] = frame
            else:
                span[key] = [frame, frame]
            # Only newer logs have the columns flicker() needs (matched, label, number).
            if "matched" in row:
                x, y, w, h = (float(row[k]) for k in ("x", "y", "w", "h"))
                # thin = drawn as a prediction: the `coasting` column, or for logs without it, a row
                # that was not matched to a detection.
                rows_by_track[row["id"]].append({
                    "frame": frame, "class": row["class"], "label": row["label"],
                    "number": row["number"],
                    "thin": row["coasting"] == "1" if "coasting" in row else row["matched"] == "0",
                    "cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h})

    print(f"== {path}")
    if not frames:
        print("   (no tracks logged)")
        return
    # The clip's length from the first to the last row's time; frames with no tracks still
    # count in nframes.
    seconds = (max(times) - min(times)) / 1000
    nframes = max(frames) - min(frames) + 1
    fps = nframes / seconds if seconds > 0 else float("nan")
    print(f"   frames {nframes}, {seconds:.1f} s, ~{fps:.1f} FPS (frames with no tracks still count)")

    by_class = defaultdict(list)
    # A track's life: frames from its first row to its last, gaps included.
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

    # Flicker needs the newer columns and a nonzero duration (the counts are per minute).
    if rows_by_track and seconds > 0:
        flicker(rows_by_track, seconds / 60)


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 1
    for path in sys.argv[1:]:
        summarize(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
