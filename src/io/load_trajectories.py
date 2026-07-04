"""Load ETH/UCY obsmat.txt files into the canonical trajectory format.

Canonical format (identical across all datasets):
    scene_id, frame, timestamp, person_id, x_m, y_m, source
All coordinates are in meters on the ground plane.
source='gt' marks ground-truth annotations (vs. 'tracked' later on).
"""
from pathlib import Path

import numpy as np
import pandas as pd

# obsmat columns: frame, ped_id, pos_x, pos_z, pos_y, v_x, v_z, v_y
# Ground plane is (pos_x, pos_y); pos_z is height and gets dropped.

ANNOTATION_DT = 0.4  # annotations are sampled at 2.5 Hz


def load_obsmat(obsmat_path: str | Path, scene_id: str) -> pd.DataFrame:
    path = Path(obsmat_path)
    raw = np.loadtxt(path)
    df = pd.DataFrame({
        "scene_id": scene_id,
        "frame": raw[:, 0].astype(int),
        "person_id": raw[:, 1].astype(int),
        "x_m": raw[:, 2],
        "y_m": raw[:, 4],
        "source": "gt",
    })
    # Annotation step is 0.4 s, but the frame step differs per scene
    # (ETH: 6, UCY: 10), so infer it from the data instead of hardcoding.
    frame0 = df["frame"].min()
    frame_step = int(np.diff(np.unique(df["frame"])).min())
    df["timestamp"] = (df["frame"] - frame0) / frame_step * ANNOTATION_DT
    return df[["scene_id", "frame", "timestamp", "person_id", "x_m", "y_m", "source"]]


def load_homography(h_path: str | Path) -> np.ndarray:
    """H maps pixels -> meters. Use its inverse for meters -> pixels."""
    return np.loadtxt(h_path)


if __name__ == "__main__":
    root = Path("../OpenTraj/datasets")
    eth = load_obsmat(root / "ETH/seq_eth/obsmat.txt", "eth")
    zara = load_obsmat(root / "UCY/zara01/obsmat.txt", "zara01")
    print("ETH:", eth.shape, "| persons:", eth.person_id.nunique())
    print(eth.head())
    print("\nZara01:", zara.shape, "| persons:", zara.person_id.nunique())
    print(zara.head())