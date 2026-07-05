"""Detect and track people in a video with YOLO11 + ByteTrack.

Outputs tracks in pixel coordinates as parquet:
    scene_id, frame, person_id, u_px, v_px, conf
(u_px, v_px) is the bottom-center of the bbox -- the foot point,
which is what gets projected onto the ground plane later.

Detector config (yolo11s, native 640, TTA) was selected by measured
recall/precision vs GT -- see docs/perception_notes.md.

Usage: python src/perception/track.py zara01
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONFIG = Path("configs/scenes.yaml")
MODEL = "yolo11s.pt"  # measured best recall on this footage (84.4% vs 73.9% for m)

# Phantom-track filter: a real pedestrian track lives >= ~1 s (25 frames
# at 25 fps) and actually goes somewhere (>= 30 px net displacement).
# Compensates for the low precision (53%) of the high-recall detector config:
# static FPs (mannequins, poles) fail the displacement test.
MIN_TRACK_LEN = 25
MIN_DISP_PX = 30


def filter_tracks(df: pd.DataFrame) -> pd.DataFrame:
    def net_displacement(t: pd.DataFrame) -> float:
        d = t[["u_px", "v_px"]].iloc[-1] - t[["u_px", "v_px"]].iloc[0]
        return float((d ** 2).sum() ** 0.5)

    length = df.groupby("person_id")["frame"].size()
    disp = df.groupby("person_id").apply(net_displacement, include_groups=False)
    keep = length.index[(length >= MIN_TRACK_LEN) & (disp >= MIN_DISP_PX)]
    return df[df.person_id.isin(keep)]


def main(scene_id: str):
    scenes = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = scenes[scene_id]
    video = Path(cfg["dir"]) / cfg["video"]

    model = YOLO(MODEL)
    rows = []
    # stream=True: process frame by frame without loading the video into RAM
    results = model.track(
        source=str(video),
        tracker="configs/bytetrack_custom.yaml",
        classes=[0],          # person class only
        conf=0.1,
        augment=True,         # TTA: +22 points recall on this footage
        device=0,             # GPU
        stream=True,
        verbose=False,
    )
    frame_idx = -1
    for frame_idx, r in enumerate(results):
        if r.boxes.id is None:
            continue
        for box, tid, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.id.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            rows.append(dict(scene_id=scene_id, frame=frame_idx,
                             person_id=int(tid),
                             u_px=float((x1 + x2) / 2),  # bottom-center =
                             v_px=float(y2),             # foot point
                             conf=float(conf)))

    df = pd.DataFrame(rows)
    n_before = df.person_id.nunique()
    df = filter_tracks(df)
    n_after = df.person_id.nunique()
    avg_len = len(df) / n_after if n_after else 0

    out = Path(f"data/processed/tracks_{scene_id}_px.parquet")
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out)
    print(f"Frames processed: {frame_idx + 1} | detections kept: {len(df)}")
    print(f"Tracks: {n_before} -> {n_after} after phantom filter "
          f"| avg track length: {avg_len:.0f} frames")
    print(f"Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    main(ap.parse_args().scene)