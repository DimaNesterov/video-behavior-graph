"""Render tracked trajectories over the source video (MVP 0 deliverable).

Usage: python scripts/render_tracks.py zara01 --start 0 --duration 60
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG = Path("configs/scenes.yaml")
TRAIL = 50  # frames of trajectory tail to draw


def main(scene_id: str, start: int, duration: int):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]
    tracks = pd.read_parquet(f"data/processed/tracks_{scene_id}_px.parquet")
    video = Path(cfg["dir"]) / cfg["video"]

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(f"outputs/tracks_{scene_id}.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    rng = np.random.default_rng(0)
    colors = {pid: tuple(int(c) for c in rng.integers(60, 255, 3))
              for pid in tracks.person_id.unique()}

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    end = start + int(duration * fps)
    for f in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        window = tracks[(tracks.frame > f - TRAIL) & (tracks.frame <= f)]
        for pid, t in window.groupby("person_id"):
            pts = t.sort_values("frame")[["u_px", "v_px"]].to_numpy().astype(int)
            for p, q in zip(pts[:-1], pts[1:]):
                cv2.line(frame, tuple(p), tuple(q), colors[pid], 2)
            cv2.circle(frame, tuple(pts[-1]), 4, colors[pid], -1)
            cv2.putText(frame, str(pid), tuple(pts[-1] + [6, -6]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[pid], 1)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--duration", type=int, default=60, help="seconds")
    args = ap.parse_args()
    main(args.scene, args.start, args.duration)