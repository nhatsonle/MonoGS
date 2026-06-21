"""Stitch the per-frame `compare/` images (GT | render) into a demo video.

The eval pipeline (utils/eval_utils.py) writes side-by-side
``[ground-truth | rendered]`` frames into ``<run>/viz/<iter>/compare/``.
This script orders them by frame index and encodes an mp4 with cv2.

Usage:
    python scripts/make_compare_video.py <compare_dir> [-o out.mp4] [--fps 20]
                                         [--no-labels]

Example:
    python scripts/make_compare_video.py \
        results/f0a37d9052_f0a37d9052/2026-06-21-14-48-42/viz/after_opt/compare \
        -o demo.mp4 --fps 20
"""

import argparse
import glob
import os
import re

import cv2


def frame_index(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("compare_dir", help="folder containing the compare/*.png frames")
    ap.add_argument("-o", "--out", default=None, help="output mp4 path")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument(
        "--no-labels",
        action="store_true",
        help="do not draw GT / Render text and the center divider",
    )
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.compare_dir, "*.png")), key=frame_index)
    if not frames:
        raise SystemExit(f"No .png frames found in {args.compare_dir}")

    out = args.out or os.path.join(
        os.path.dirname(args.compare_dir.rstrip("/")), "compare_demo.mp4"
    )

    h, w = cv2.imread(frames[0]).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, args.fps, (w, h))

    for i, f in enumerate(frames):
        img = cv2.imread(f)
        if img is None or img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h)) if img is not None else None
            if img is None:
                continue
        if not args.no_labels:
            cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
            cv2.putText(img, "Ground truth", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, "Ground truth", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "Rendered", (w // 2 + 12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, "Rendered", (w // 2 + 12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            label = f"frame {frame_index(f):06d}"
            cv2.putText(img, label, (12, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (12, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    print(f"Wrote {len(frames)} frames -> {out} ({w}x{h} @ {args.fps} fps)")


if __name__ == "__main__":
    main()
