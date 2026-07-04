"""Sanity check: project GT trajectories (meters) onto a video frame via homography.

Usage:
    python scripts/sanity_overlay.py zara01
    python scripts/sanity_overlay.py eth
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io.load_trajectories import load_obsmat, load_homography

CONFIG = Path("configs/scenes.yaml")


def world_to_image(xy_m: np.ndarray, H: np.ndarray, swap_output: bool) -> np.ndarray:
    """Project world coordinates (meters) to image pixels.

    H maps pixels -> meters, so we use its inverse.
    Some scenes (UCY) yield (row, col) order; swap_output flips it to
    (col, row) as expected by OpenCV drawing functions.
    """
    H_inv = np.linalg.inv(H)
    pts = np.hstack([xy_m, np.ones((len(xy_m), 1))])  # to homogeneous
    px = (H_inv @ pts.T).T
    px = px[:, :2] / px[:, 2:3]  # normalize
    if swap_output:
        px = px[:, [1, 0]]
    return px


def main(scene_id: str):
    scenes = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if scene_id not in scenes:
        sys.exit(f"Scene '{scene_id}' not found in {CONFIG}. Available: {list(scenes)}")
    cfg = scenes[scene_id]
    scene_dir = Path(cfg["dir"])

    df = load_obsmat(scene_dir / cfg["obsmat"], scene_id)
    H = load_homography(scene_dir / cfg["homography"])

    cap = cv2.VideoCapture(str(scene_dir / cfg["video"]))
    ok, frame = cap.read()
    cap.release()
    assert ok, "Failed to read a video frame"

    h, w = frame.shape[:2]
    rng = np.random.default_rng(0)
    for _, traj in df.groupby("person_id"):
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        px = world_to_image(traj[["x_m", "y_m"]].to_numpy(), H, cfg["swap_px_output"])
        pts = px.astype(int)
        # drop points outside the frame
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
        for p, q in zip(pts[:-1], pts[1:]):
            cv2.line(frame, tuple(p), tuple(q), color, 1)

    out = Path(f"outputs/sanity_{scene_id}.png")
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="Scene ID from configs/scenes.yaml")
    main(ap.parse_args().scene)