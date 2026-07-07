"""Render the dynamic graph over the source video: person nodes with
their live edges to objects/entries. The first artifact where the graph
visibly breathes.

Usage: python scripts/render_dynamic_graph.py zara01 --start 1000 --duration 45
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io.load_trajectories import load_homography
from scripts.sanity_overlay import world_to_image

CONFIG = Path("configs/scenes.yaml")

EDGE_COLORS = {          # BGR
    "near": (0, 200, 255),
    "heading_to": (255, 150, 0),
}


def main(scene_id: str, start: int, duration: int, source: str):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]
    H = load_homography(Path(cfg["dir"]) / cfg["homography"])
    swap, flip = cfg["swap_px_output"], cfg.get("flip_px_x", False)

    frames = json.loads(Path(
        f"data/processed/graph_{scene_id}_dynamic_{source}.json"
    ).read_text(encoding="utf-8"))
    by_frame = {fr["frame"]: fr for fr in frames}
    static = json.loads(Path(
        f"data/processed/graph_{scene_id}_static.json").read_text(encoding="utf-8"))
    static_pos = {}
    for n in static["nodes"]:
        p = n.get("pos_m") or n.get("location_m")
        if p is not None:
            static_pos[n["id"]] = np.array(p)

    video = Path(cfg["dir"]) / cfg["video"]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(f"outputs/dynamic_graph_{scene_id}_{source}.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    def to_px(pt_m):
        return tuple(world_to_image(np.array([pt_m]), H, swap, flip, w)
                     .astype(int)[0])

    grid_keys = sorted(by_frame.keys())
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    end = start + int(duration * fps)
    for f in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        # nearest grid snapshot at or before f
        k = max((g for g in grid_keys if g <= f), default=None)
        if k is None:
            writer.write(frame)
            continue
        snap = by_frame[k]

        for n in static["nodes"]:
            p = n.get("pos_m") or n.get("location_m")
            if p is not None:
                cv2.circle(frame, to_px(p), 6, (0, 200, 255), -1)

        pos_by_person = {p["id"]: p["pos_m"] for p in snap["persons"]}
        for e in snap["edges"]:
            if e["type"] not in EDGE_COLORS:
                continue
            src = pos_by_person.get(e["source"])
            tgt = static_pos.get(e["target"])
            if src is None or tgt is None:
                continue
            color = EDGE_COLORS[e["type"]]
            thick = 2 if e.get("approaching") else 1
            cv2.line(frame, to_px(src), to_px(tgt), color, thick)

        for p in snap["persons"]:
            color = (0, 0, 255) if p["stopped"] else (0, 255, 0)
            cv2.circle(frame, to_px(p["pos_m"]), 6, color, -1)
            cv2.putText(frame, f'{p["speed"]:.1f}', 
                        tuple(np.array(to_px(p["pos_m"])) + [8, -8]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--duration", type=int, default=45, help="seconds")
    ap.add_argument("--source", default="gt", choices=["gt", "tracked"])
    args = ap.parse_args()
    main(args.scene, args.start, args.duration, args.source)