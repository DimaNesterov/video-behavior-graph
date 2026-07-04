"""Round-trip test: meters -> pixels -> meters must be lossless.

Note: swap_output=False on purpose. The swap only reorders columns for
OpenCV drawing; the round-trip checks invertibility of the projection
itself, and swapped columns would break the inverse mapping.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io.load_trajectories import load_obsmat, load_homography
from scripts.sanity_overlay import world_to_image


def test_roundtrip_zara01():
    root = Path("../OpenTraj/datasets/UCY/zara01")
    df = load_obsmat(root / "obsmat.txt", "zara01")
    H = load_homography(root / "H.txt")

    xy = df[["x_m", "y_m"]].to_numpy()[:200]
    px = world_to_image(xy, H, swap_output=False)   # meters -> pixels
    back = np.hstack([px, np.ones((len(px), 1))])   # pixels -> meters
    back = (H @ back.T).T
    back = back[:, :2] / back[:, 2:3]

    err = np.abs(back - xy).max()
    assert err < 1e-6, f"Round-trip error too large: {err}"


if __name__ == "__main__":
    test_roundtrip_zara01()
    print("Round-trip OK")