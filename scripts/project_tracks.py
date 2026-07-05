"""Project tracked pixel positions into world meters, resample onto the 0.4 s
annotation grid, and compare against GT.

The pixel->meters transform is the exact inverse of world_to_image:
undo the x-flip, undo the (col,row) swap, then apply H directly.

Usage: python scripts/project_tracks.py zara01
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io.load_trajectories import load_obsmat, load_homography, ANNOTATION_DT
from scripts.eval_baseline import build_windows

CONFIG = Path("configs/scenes.yaml")
MATCH_RADIUS_M = 1.0
SMOOTH_WIN = 5  # frames; foot-point jitter is noisy at 25 fps


def image_to_world(uv: np.ndarray, H: np.ndarray, swap_input: bool,
                   flip_x: bool, frame_w: int) -> np.ndarray:
    uv = uv.astype(float).copy()
    if flip_x:
        uv[:, 0] = uv[:, 0] - frame_w   # undo the mirror used for drawing
    if swap_input:
        uv = uv[:, [1, 0]]              # back to H-native (row, col) order
    pts = np.hstack([uv, np.ones((len(uv), 1))])
    w = (H @ pts.T).T
    return w[:, :2] / w[:, 2:3]


def main(scene_id: str):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]
    scene_dir = Path(cfg["dir"])
    tracks = pd.read_parquet(f"data/processed/tracks_{scene_id}_px.parquet")
    gt = load_obsmat(scene_dir / cfg["obsmat"], scene_id)
    H = load_homography(scene_dir / cfg["homography"])

    cap = cv2.VideoCapture(str(scene_dir / cfg["video"]))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    xy = image_to_world(tracks[["u_px", "v_px"]].to_numpy(), H,
                        cfg["swap_px_output"], cfg.get("flip_px_x", False), frame_w)
    tracks["x_m"], tracks["y_m"] = xy[:, 0], xy[:, 1]

    # Resample every track onto the GT annotation grid (step = 10 frames here).
    step = int(np.diff(np.unique(gt.frame)).min())
    frame0 = int(gt.frame.min())
    grid = np.arange(frame0, int(tracks.frame.max()) + 1, step)

    rows = []
    for pid, t in tracks.sort_values("frame").groupby("person_id"):
        t = t.drop_duplicates("frame")
        if len(t) < SMOOTH_WIN:
            continue
        xs = t.x_m.rolling(SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy()
        ys = t.y_m.rolling(SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy()
        f = t.frame.to_numpy()
        g = grid[(grid >= f.min()) & (grid <= f.max())]
        if len(g) < 2:
            continue
        for gf, x, y in zip(g, np.interp(g, f, xs), np.interp(g, f, ys)):
            rows.append(dict(scene_id=scene_id, frame=int(gf),
                             timestamp=(gf - frame0) / step * ANNOTATION_DT,
                             person_id=int(pid), x_m=float(x), y_m=float(y),
                             source="tracked"))
    tdf = pd.DataFrame(rows)
    out = Path(f"data/processed/tracks_{scene_id}_world.parquet")
    tdf.to_parquet(out)

    # --- Compare with GT ---
    # Coverage: share of GT observations that have a tracked point within 1 m
    # on the same frame. Position error: mean distance over those matches.
    tp, errs = 0, []
    common = sorted(set(gt.frame) & set(tdf.frame))
    gt_c = gt[gt.frame.isin(common)]
    for f, g in gt_c.groupby("frame"):
        cand = tdf[tdf.frame == f][["x_m", "y_m"]].to_numpy()
        for _, row in g.iterrows():
            if not len(cand):
                continue
            d = np.linalg.norm(cand - row[["x_m", "y_m"]].to_numpy(dtype=float), axis=1)
            if d.min() < MATCH_RADIUS_M:
                tp += 1
                errs.append(d.min())
    coverage = tp / len(gt_c)
    win_tracked = len(build_windows(tdf))
    win_gt = len(build_windows(gt))

    print(f"Tracked: {tdf.person_id.nunique()} tracks, {len(tdf)} resampled points")
    print(f"Coverage vs GT: {coverage:.1%} of GT observations matched within {MATCH_RADIUS_M} m")
    print(f"Position error on matches: mean {np.mean(errs):.2f} m, median {np.median(errs):.2f} m")
    print(f"Valid 8+12 windows: tracked {win_tracked} vs GT {win_gt}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    main(ap.parse_args().scene)