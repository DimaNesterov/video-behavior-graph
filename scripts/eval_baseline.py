"""Evaluate the constant velocity baseline on ETH/UCY scenes.

Usage: python scripts/eval_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import yaml

import csv
import subprocess
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io.load_trajectories import load_obsmat
from src.prediction.baselines import constant_velocity
from src.prediction.metrics import ade, fde

CONFIG = Path("configs/scenes.yaml")
OBS_LEN, PRED_LEN = 8, 12  # 3.2 s observed, 4.8 s predicted (standard protocol)


def build_windows(df, obs_len=OBS_LEN, pred_len=PRED_LEN):
    """Slice each person's trajectory into (obs, future) windows.

    Only windows with strictly consecutive annotation steps are kept,
    so gaps (occlusions, re-entries) never produce corrupted samples.
    """
    total = obs_len + pred_len
    windows = []
    for _, traj in df.groupby("person_id"):
        traj = traj.sort_values("frame")
        xy = traj[["x_m", "y_m"]].to_numpy()
        frames = traj["frame"].to_numpy()
        if len(xy) < total:
            continue
        step = int(np.diff(np.unique(df["frame"])).min())
        for i in range(len(xy) - total + 1):
            if frames[i + total - 1] - frames[i] == (total - 1) * step:
                windows.append((xy[i:i + obs_len], xy[i + obs_len:i + total]))
    return windows


def main():
    scenes = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    rows = []
    print(f"{'scene':<10} {'windows':>8} {'ADE':>7} {'FDE':>7}")
    for scene_id, cfg in scenes.items():
        df = load_obsmat(Path(cfg["dir"]) / cfg["obsmat"], scene_id)
        windows = build_windows(df)
        if not windows:
            print(f"{scene_id:<10} {'0':>8}   no valid windows")
            continue
        ades, fdes = [], []
        for obs, future in windows:
            pred = constant_velocity(obs, PRED_LEN)
            ades.append(ade(pred, future))
            fdes.append(fde(pred, future))
        row = dict(date=date.today().isoformat(), commit=commit, model="const_velocity",
                   scene=scene_id, windows=len(windows), obs_len=OBS_LEN, pred_len=PRED_LEN,
                   ade=round(float(np.mean(ades)), 3), fde=round(float(np.mean(fdes)), 3))
        rows.append(row)
        print(f"{scene_id:<10} {row['windows']:>8} {row['ade']:>7.3f} {row['fde']:>7.3f}")

    out = Path("results/baselines.csv")
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nAppended {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()