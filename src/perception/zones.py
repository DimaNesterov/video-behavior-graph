"""Zones from trajectories: tiered activity area + entry/exit nodes.

Activity tiers (spec section 9.3, high/low flow zones): occupancy grid
(0.25 m cells) over all trajectory points, split into three tiers by
visit-count quantiles -- adaptive per scene, no absolute thresholds.
  tier 3 = high activity, tier 2 = medium, tier 1 = low, 0 = not walkable.
The binary walkable mask (union of tiers) is kept for backward compatibility.

Entry/exit nodes, two kinds:
  1) building entries -- endpoints near a detected opening (door/doorway/
     entrance/gate from the verified object layer) are peeled off BEFORE
     clustering and become their own nodes, linked to the object.
     Semantics splits what geometry cannot: door traffic and sidewalk
     traffic form one dense chain that DBSCAN would merge.
  2) boundary exits -- DBSCAN clusters of the remaining endpoints.
     Weak clusters survive only near the frame boundary (real scene edge);
     weak interior clusters are annotation-loss artifacts and get dropped.

Usage: python src/perception/zones.py zara01
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.ndimage import binary_closing, binary_dilation
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.io.load_trajectories import load_obsmat, load_homography
from scripts.sanity_overlay import world_to_image
from scripts.project_tracks import image_to_world

CONFIG = Path("configs/scenes.yaml")
CELL_M = 0.25          # grid cell size, meters
DBSCAN_EPS_M = 1.5     # entry/exit cluster radius
DBSCAN_MIN_PTS = 5
MIN_USAGE = 0.05       # boundary exit must hold >= 5% of endpoints (or sit at frame edge)
OPENING_TYPES = {"door", "doorway", "entrance", "gate"}
NEAR_OPENING_M = 2.5   # endpoint-to-opening radius for building entries
TIER_QUANTILES = (0.33, 0.66)  # tier splits over visited cells


def main(scene_id: str):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]
    scene_dir = Path(cfg["dir"])
    df = load_obsmat(scene_dir / cfg["obsmat"], scene_id)
    H = load_homography(scene_dir / cfg["homography"])
    swap = cfg["swap_px_output"]
    flip = cfg.get("flip_px_x", False)

    # ---------------- Activity grid and tiers ----------------
    xy = df[["x_m", "y_m"]].to_numpy()
    x0, y0 = xy.min(0) - 1.0
    x1, y1 = xy.max(0) + 1.0
    nx = int(np.ceil((x1 - x0) / CELL_M))
    ny = int(np.ceil((y1 - y0) / CELL_M))

    ix = ((xy[:, 0] - x0) / CELL_M).astype(int).clip(0, nx - 1)
    iy = ((xy[:, 1] - y0) / CELL_M).astype(int).clip(0, ny - 1)
    grid = np.zeros((ny, nx), dtype=int)
    np.add.at(grid, (iy, ix), 1)

    visited = grid[grid > 0]
    q_low, q_high = np.quantile(visited, TIER_QUANTILES)
    tiers = np.zeros_like(grid)
    tiers[grid >= 1] = 1                  # low activity
    tiers[grid > q_low] = 2               # medium
    tiers[grid > q_high] = 3              # high
    # close small gaps in the union without inventing new tiers
    closed = binary_closing(tiers > 0, iterations=2)
    tiers = np.where(closed & (tiers == 0), 1, tiers)

    walkable = tiers > 0
    walkable = binary_dilation(walkable, iterations=1)

    # ---------------- Trajectory endpoints ----------------
    ends = []
    for _, t in df.sort_values("frame").groupby("person_id"):
        p = t[["x_m", "y_m"]].to_numpy()
        ends.append(p[0])
        ends.append(p[-1])
    ends = np.array(ends)
    n_ends_total = len(ends)

    # ---------------- Building entries (semantic, pre-clustering) ----------------
    openings_m = []
    ver_path = Path(f"data/processed/objects_{scene_id}_verified.json")
    bg_path = f"data/processed/background_{scene_id}.png"
    if ver_path.exists():
        frame_w = cv2.imread(bg_path).shape[1]
        for o in json.loads(ver_path.read_text(encoding="utf-8")):
            if o["type"] in OPENING_TYPES:
                bx1, by1, bx2, by2 = o["bbox"]
                foot = np.array([[(bx1 + bx2) / 2, by2]])
                pos = image_to_world(foot, H, swap, flip, frame_w)[0]
                openings_m.append((o["id"], pos))

    opening_entries = []
    assigned = np.zeros(len(ends), dtype=bool)
    for oid, om in openings_m:
        d = np.linalg.norm(ends - om, axis=1)
        mask = (d < NEAR_OPENING_M) & ~assigned
        if mask.sum() >= DBSCAN_MIN_PTS:
            pts = ends[mask]
            opening_entries.append(dict(
                id=f"entry_exit_{oid}",
                type="entry_exit",
                subtype="building_entry",
                linked_object=oid,
                location_m=[round(float(v), 2) for v in pts.mean(0)],
                usage_count=int(mask.sum()),
                usage_frequency=round(float(mask.sum() / n_ends_total), 3),
            ))
            assigned |= mask

    # ---------------- Boundary exits (geometric, DBSCAN) ----------------
    rest = ends[~assigned]
    labels = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_PTS).fit_predict(rest)

    entries = []
    for k in sorted(set(labels) - {-1}):
        pts = rest[labels == k]
        entries.append(dict(
            id=f"entry_exit_{k:02d}",
            type="entry_exit",
            subtype="boundary_exit",
            location_m=[round(float(v), 2) for v in pts.mean(0)],
            usage_count=int(len(pts)),
            usage_frequency=round(float(len(pts) / n_ends_total), 3),
        ))

    margin_x = (x1 - x0) * 0.12
    margin_y = (y1 - y0) * 0.12

    def near_boundary(loc):
        return (loc[0] < x0 + margin_x or loc[0] > x1 - margin_x
                or loc[1] < y0 + margin_y or loc[1] > y1 - margin_y)

    entries = [e for e in entries
               if e["usage_frequency"] >= MIN_USAGE or near_boundary(e["location_m"])]
    entries = opening_entries + entries

    # ---------------- Save ----------------
    np.savez(f"data/processed/zones_{scene_id}.npz",
             walkable=walkable, tiers=tiers, origin=[x0, y0], cell=CELL_M)
    Path(f"data/processed/entries_{scene_id}.json").write_text(
        json.dumps(entries, indent=2), encoding="utf-8")

    # ---------------- Overlay ----------------
    bg = cv2.imread(bg_path)
    h, w = bg.shape[:2]
    TIER_COLORS = {1: (80, 160, 80), 2: (0, 220, 120), 3: (0, 255, 0)}
    overlay = bg.copy()
    for tier, color in TIER_COLORS.items():
        ty, tx = np.where(tiers == tier)
        if not len(ty):
            continue
        cells_m = np.stack([x0 + (tx + 0.5) * CELL_M, y0 + (ty + 0.5) * CELL_M], 1)
        px = world_to_image(cells_m, H, swap, flip, w).astype(int)
        ok = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        for p in px[ok]:
            cv2.circle(overlay, tuple(p), 3, color, -1)
    bg = cv2.addWeighted(overlay, 0.35, bg, 0.65, 0)
    for e in entries:
        p = world_to_image(np.array([e["location_m"]]), H, swap, flip, w).astype(int)[0]
        color = (255, 100, 0) if e.get("subtype") == "building_entry" else (0, 100, 255)
        cv2.circle(bg, tuple(p), 12, color, 3)
        cv2.putText(bg, e["id"], (p[0] + 14, p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    out = f"outputs/zones_{scene_id}.png"
    cv2.imwrite(out, bg)

    print(f"Walkable cells: {int(walkable.sum())} "
          f"({walkable.sum() * CELL_M**2:.0f} m^2) | "
          f"high {int((tiers == 3).sum())} / med {int((tiers == 2).sum())} / "
          f"low {int((tiers == 1).sum())}")
    print(f"Entry/exit nodes: {len(entries)}")
    for e in entries:
        print(f'  {e["id"]} [{e.get("subtype", "-")}]: at {e["location_m"]}, '
              f'usage {e["usage_frequency"]}')
    print(f"Saved: zones npz, entries json, {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    main(ap.parse_args().scene)