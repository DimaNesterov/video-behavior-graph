"""Dump sample frames with raw YOLO detections to eyeball what gets missed.

Usage: python scripts/debug_detections.py zara01
"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG = Path("configs/scenes.yaml")
SAMPLE_FRAMES = [500, 2000, 4000, 6000, 8000]


def main(scene_id: str):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]
    video = str(Path(cfg["dir"]) / cfg["video"])
    model = YOLO("yolo11s.pt")

    cap = cv2.VideoCapture(video)
    out_dir = Path("outputs/debug_det")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in SAMPLE_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        # low conf on purpose: we want to see everything the model considers
        r = model.predict(frame, classes=[0], conf=0.1, device=0, verbose=False)[0]
        annotated = r.plot()  # draws boxes with confidences
        cv2.imwrite(str(out_dir / f"{scene_id}_f{f}.png"), annotated)
        print(f"frame {f}: {len(r.boxes)} detections")
    cap.release()
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    main(ap.parse_args().scene)