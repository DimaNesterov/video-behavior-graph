"""Dynamic graph layer: person nodes + person->scene edges on the 0.4 s grid.

For every annotation-grid timestep, each person present in the scene becomes
a person node with kinematic attributes (speed, heading, stop score), and
gets edges to static graph elements:
  person --near--   object      (distance < NEAR_M, with distance_m,
                                 angle_alignment: is the person heading at it)
  person --heading_to-- entry   (aligned with an entry/exit and approaching)
  person --in_zone-- zone       (activity tier at the person's cell)

Output: per-timestep graph snapshots in one JSON (list of frames), built on
canonical trajectories -- source='gt' by default, 'tracked' via --source.

Usage: python src/graph/build_dynamic.py zara01
       python src/graph/build_dynamic.py zara01 --source tracked
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.io.load_trajectories import load_obsmat, ANNOTATION_DT

CONFIG = Path("configs/scenes.yaml")
NEAR_M = 4.0            # person-object proximity radius
STOP_SPEED = 0.3        # m/s below which a person counts as stopped
ALIGN_COS = 0.7         # heading alignment threshold (cos of ~45 deg)
HEADING_MAX_M = 8.0     # heading_to edges only within this range


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def main(scene_id: str, source: str):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]

    if source == "gt":
        df = load_obsmat(Path(cfg["dir"]) / cfg["obsmat"], scene_id)
    else:
        df = pd.read_parquet(f"data/processed/tracks_{scene_id}_world.parquet")

    static = json.loads(Path(
        f"data/processed/graph_{scene_id}_static.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in static["nodes"]}
    objects = [(nid, np.array(n["pos_m"])) for nid, n in nodes.items()
               if n.get("pos_m") is not None]
    entries = [(nid, np.array(n["location_m"])) for nid, n in nodes.items()
               if n["type"] == "entry_exit"]

    z = np.load(f"data/processed/zones_{scene_id}.npz")
    tiers, (zx0, zy0), cell = z["tiers"], z["origin"], float(z["cell"])

    def tier_at(pos):
        cx = int((pos[0] - zx0) / cell)
        cy = int((pos[1] - zy0) / cell)
        if 0 <= cy < tiers.shape[0] and 0 <= cx < tiers.shape[1]:
            return int(tiers[cy, cx])
        return 0

    frames_out = []
    df = df.sort_values(["frame", "person_id"])
    grid_frames = sorted(df.frame.unique())
    prev_pos = {}  # person_id -> position at previous grid step

    for f in grid_frames:
        snap = df[df.frame == f]
        persons, edges = [], []
        for _, row in snap.iterrows():
            pid = int(row.person_id)
            pos = np.array([row.x_m, row.y_m])
            vel = ((pos - prev_pos[pid]) / ANNOTATION_DT
                   if pid in prev_pos else np.zeros(2))
            speed = float(np.linalg.norm(vel))
            heading = unit(vel)
            persons.append(dict(
                id=f"person_{pid}",
                type="person",
                pos_m=[round(float(v), 2) for v in pos],
                speed=round(speed, 2),
                heading=[round(float(v), 3) for v in heading],
                stopped=bool(speed < STOP_SPEED),
            ))
            prev_pos[pid] = pos

            for oid, opos in objects:
                d = float(np.linalg.norm(opos - pos))
                if d < NEAR_M:
                    align = float(np.dot(heading, unit(opos - pos)))
                    edges.append(dict(
                        source=f"person_{pid}", target=oid, type="near",
                        distance_m=round(d, 2),
                        angle_alignment=round(align, 2),
                        approaching=bool(align > ALIGN_COS and speed >= STOP_SPEED),
                    ))
            for eid, epos in entries:
                to_e = epos - pos
                d = float(np.linalg.norm(to_e))
                align = float(np.dot(heading, unit(to_e)))
                if align > ALIGN_COS and speed >= STOP_SPEED and d < HEADING_MAX_M:
                    edges.append(dict(
                        source=f"person_{pid}", target=eid, type="heading_to",
                        distance_m=round(d, 2),
                        angle_alignment=round(align, 2),
                    ))
            edges.append(dict(
                source=f"person_{pid}", target="zone_00", type="in_zone",
                activity_tier=tier_at(pos),
            ))

        frames_out.append(dict(
            frame=int(f),
            timestamp=round(float(snap.timestamp.iloc[0]), 2),
            persons=persons,
            edges=edges,
        ))

    out = Path(f"data/processed/graph_{scene_id}_dynamic_{source}.json")
    out.write_text(json.dumps(frames_out), encoding="utf-8")

    n_persons = sum(len(fr["persons"]) for fr in frames_out)
    n_edges = sum(len(fr["edges"]) for fr in frames_out)
    n_near = sum(1 for fr in frames_out for e in fr["edges"] if e["type"] == "near")
    n_head = sum(1 for fr in frames_out for e in fr["edges"] if e["type"] == "heading_to")
    print(f"Timesteps: {len(frames_out)} | person-observations: {n_persons}")
    print(f"Edges: {n_edges} (near: {n_near}, heading_to: {n_head})")
    print(f"Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--source", default="gt", choices=["gt", "tracked"])
    main(ap.parse_args().scene, ap.parse_args().source)