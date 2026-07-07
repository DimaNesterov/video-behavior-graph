"""Extract a clean background frame via per-pixel temporal median.

People move; the background does not. The median over ~100 random frames
erases every pedestrian and leaves an empty scene -- the input for
static object detection (Phase 2).

Usage: python src/perception/background.py zara01
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONFIG = Path("configs/scenes.yaml")
N_SAMPLES = 101  # odd -> true median element per pixel


def main(scene_id: str):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]
    video = Path(cfg["dir"]) / cfg["video"]

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, N_SAMPLES).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    out = Path(f"data/processed/background_{scene_id}.png")
    cv2.imwrite(str(out), bg)
    print(f"Sampled {len(frames)} frames -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    main(ap.parse_args().scene)