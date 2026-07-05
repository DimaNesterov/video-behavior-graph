"""Detection recall/precision vs GT on annotated frames.

Projects GT world coords to pixels, matches YOLO foot points within a radius.
Usage: python scripts/eval_detection.py zara01
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io.load_trajectories import load_obsmat, load_homography
from scripts.sanity_overlay import world_to_image

CONFIG = Path("configs/scenes.yaml")
MATCH_RADIUS_PX = 40
SAMPLE_EVERY = 20  # every 20th annotated frame -> ~25 frames for zara01

def apply_clahe(img):
    """Adaptive contrast enhancement: lifts people out of deep shadows."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def deinterlace(img):
    """Field-based deinterlace: drop odd lines, resize back. Removes combing."""
    h, w = img.shape[:2]
    return cv2.resize(img[::2], (w, h), interpolation=cv2.INTER_LINEAR)


def main(scene_id, use_clahe, use_deint, tta, imgsz, model_name):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]
    scene_dir = Path(cfg["dir"])
    df = load_obsmat(scene_dir / cfg["obsmat"], scene_id)
    H = load_homography(scene_dir / cfg["homography"])
    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(scene_dir / cfg["video"]))
    frames = sorted(df.frame.unique())[::SAMPLE_EVERY]
    tp = fn = fp = 0
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        if use_clahe:
            img = apply_clahe(img)
        if use_deint:
            img = deinterlace(img)
        gt = df[df.frame == f]
        gt_px = world_to_image(gt[["x_m", "y_m"]].to_numpy(), H,
                               cfg["swap_px_output"], cfg.get("flip_px_x", False),
                               img.shape[1])
        r = model.predict(img, classes=[0], conf=0.1, imgsz=imgsz,
                          augment=tta, device=0, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        det_px = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1) \
            if len(boxes) else np.zeros((0, 2))

        matched_det = set()
        for g in gt_px:
            d = np.linalg.norm(det_px - g, axis=1) if len(det_px) else np.array([])
            hit = np.where(d < MATCH_RADIUS_PX)[0]
            hit = [i for i in hit if i not in matched_det]
            if hit:
                tp += 1
                matched_det.add(hit[np.argmin(d[hit])] if len(hit) > 1 else hit[0])
            else:
                fn += 1
        fp += len(det_px) - len(matched_det)
    cap.release()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    print(f"{scene_id}: frames={len(frames)} GT={tp + fn} "
          f"recall={recall:.2%} precision={precision:.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--deint", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--model", default="yolo11m.pt")
    args = ap.parse_args()
    main(args.scene, args.clahe, args.deint, args.tta, args.imgsz, args.model)