"""Sanity-check: GT-траектории (метры) -> пиксели через H_inv -> поверх кадра видео.
Если траектории лягут на тротуар и на людей — вся координатная цепочка верна."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io.load_trajectories import load_obsmat, load_homography

SCENE_DIR = Path("../OpenTraj/datasets/UCY/zara01")
OUT = Path("outputs/sanity_zara01.png")

def world_to_image(xy_m: np.ndarray, H: np.ndarray) -> np.ndarray:
    """(N,2) метры -> (N,2) пиксели. H переводит пиксели->метры, поэтому берём обратную."""
    H_inv = np.linalg.inv(H)
    pts = np.hstack([xy_m, np.ones((len(xy_m), 1))])          # -> гомогенные
    px = (H_inv @ pts.T).T
    px = px[:, :2] / px[:, 2:3]                                # нормировка
    return px

def main():
    df = load_obsmat(SCENE_DIR / "obsmat.txt", "zara01")
    H = load_homography(SCENE_DIR / "H.txt")

    cap = cv2.VideoCapture(str(SCENE_DIR / "video.avi"))
    ok, frame = cap.read()
    cap.release()
    assert ok, "Не удалось прочитать кадр видео"

    rng = np.random.default_rng(0)
    for pid, traj in df.groupby("person_id"):
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        px = world_to_image(traj[["x_m", "y_m"]].to_numpy(), H)
        px = px[:, [1, 0]]  # UCY: проекция даёт (row, col) -> меняем на (col, row) для OpenCV
        pts = px.astype(int)
        # отбрасываем точки за пределами кадра
        h, w = frame.shape[:2]
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
        for p, q in zip(pts[:-1], pts[1:]):
            cv2.line(frame, tuple(p), tuple(q), color, 1)

    OUT.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT), frame)
    print(f"Сохранено: {OUT.resolve()}")

if __name__ == "__main__":
    main()