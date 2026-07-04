"""Загрузка ETH/UCY obsmat.txt в канонический формат.

Канонический формат (одинаковый для всех датасетов):
    scene_id, frame, timestamp, person_id, x_m, y_m, source
Всё в метрах на плоскости земли. source='gt' для ground-truth аннотаций.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# obsmat: frame, ped_id, pos_x, pos_z, pos_y, v_x, v_z, v_y
# Плоскость земли — это pos_x и pos_y. pos_z (высота) отбрасывается.
FPS = 2.5  # аннотации на 2.5 Гц, шаг 0.4 с

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
        # шаг аннотации = 0.4 с; шаг кадров разный по сценам (ETH: 6, UCY: 10),
    # поэтому выводим его из данных, а не хардкодим
    frame0 = df["frame"].min()
    frame_step = int(np.diff(np.unique(df["frame"])).min())
    df["timestamp"] = (df["frame"] - frame0) / frame_step * 0.4
    return df[["scene_id", "frame", "timestamp", "person_id", "x_m", "y_m", "source"]]


def load_homography(h_path: str | Path) -> np.ndarray:
    """H переводит пиксели -> метры. H_inv (обратная) переводит метры -> пиксели."""
    return np.loadtxt(h_path)


if __name__ == "__main__":
    root = Path("../OpenTraj/datasets")
    eth = load_obsmat(root / "ETH/seq_eth/obsmat.txt", "eth")
    zara = load_obsmat(root / "UCY/zara01/obsmat.txt", "zara01")
    print("ETH:", eth.shape, "| людей:", eth.person_id.nunique())
    print(eth.head())
    print("\nZara01:", zara.shape, "| людей:", zara.person_id.nunique())
    print(zara.head())