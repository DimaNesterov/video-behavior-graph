"""Assemble the static scene graph: objects + walkable zone + entries.

Edges:
  object --near-- object          (ground distance < NEAR_M)
  object --adjacent_to-- zone     (distance to nearest walkable cell < ADJ_M)
  entry  --connected_to-- zone

Usage: python src/graph/build_static.py zara01
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.io.load_trajectories import load_homography
from src.graph.schema import ObjectNode, ZoneNode, EntryExitNode, Edge
from scripts.project_tracks import image_to_world
from scripts.sanity_overlay import world_to_image

CONFIG = Path("configs/scenes.yaml")
NEAR_M = 4.0
ADJ_M = 1.0


def main(scene_id: str):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]
    H = load_homography(Path(cfg["dir"]) / cfg["homography"])
    bg = cv2.imread(f"data/processed/background_{scene_id}.png")
    h, w = bg.shape[:2]
    swap, flip = cfg["swap_px_output"], cfg.get("flip_px_x", False)

    objects_raw = json.loads(Path(
        f"data/processed/objects_{scene_id}_verified.json").read_text(encoding="utf-8"))
    manual_path = Path(f"data/manual/{scene_id}_objects.json")
    if manual_path.exists():
        manual = json.loads(manual_path.read_text(encoding="utf-8"))
        objects_raw += manual
        print(f"Manual overlay: +{len(manual)} objects")
    entries_raw = json.loads(Path(
        f"data/processed/entries_{scene_id}.json").read_text(encoding="utf-8"))
    z = np.load(f"data/processed/zones_{scene_id}.npz")
    walkable, (x0, y0), cell = z["walkable"], z["origin"], float(z["cell"])

    G = nx.Graph()

    # Zone node
    zone = ZoneNode(id="zone_00", type="walkable_area",
                    area_m2=float(walkable.sum() * cell ** 2), cell_m=cell)
    G.add_node(zone.id, **zone.model_dump())

    # Object nodes: ground position = bbox bottom-center through H
    cy, cx = np.where(walkable)
    cells_m = np.stack([x0 + (cx + 0.5) * cell, y0 + (cy + 0.5) * cell], 1)
    for o in objects_raw:
        x1, y1, x2, y2 = o["bbox"]
        foot = np.array([[(x1 + x2) / 2, y2]])
        pos = image_to_world(foot, H, swap, flip, w)[0]
        node = ObjectNode(id=o["id"], type=o["type"], bbox_px=o["bbox"],
                          pos_m=[round(float(v), 2) for v in pos],
                          confidence=o["confidence"], source=o["source"])
        G.add_node(node.id, **node.model_dump())
        d_zone = float(np.linalg.norm(cells_m - pos, axis=1).min())
        if d_zone < ADJ_M:
            e = Edge(source=node.id, target=zone.id, type="adjacent_to",
                     distance_m=round(d_zone, 2))
            G.add_edge(e.source, e.target, **e.model_dump())

    # Entry/exit nodes
    for en in entries_raw:
        node = EntryExitNode(id=en["id"], type="entry_exit",
                             subtype=en.get("subtype", "boundary_exit"),
                             linked_object=en.get("linked_object"),
                             location_m=en["location_m"],
                             usage_frequency=en["usage_frequency"])
        G.add_node(node.id, **node.model_dump())
        e = Edge(source=node.id, target=zone.id, type="connected_to")
        G.add_edge(e.source, e.target, **e.model_dump())

    # object -- near -- object
    objs = [(n, d) for n, d in G.nodes(data=True) if d.get("pos_m")]
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = np.array(objs[i][1]["pos_m"]), np.array(objs[j][1]["pos_m"])
            d = float(np.linalg.norm(a - b))
            if d < NEAR_M:
                e = Edge(source=objs[i][0], target=objs[j][0], type="near",
                         distance_m=round(d, 2))
                G.add_edge(e.source, e.target, **e.model_dump())

    out = Path(f"data/processed/graph_{scene_id}_static.json")
    out.write_text(json.dumps(nx.node_link_data(G), indent=2), encoding="utf-8")

    # Viz: nodes + edges over background
    def to_px(pt_m):
        return world_to_image(np.array([pt_m]), H, swap, flip, w).astype(int)[0]

    for u, v, d in G.edges(data=True):
        pu = G.nodes[u].get("pos_m") or G.nodes[u].get("location_m")
        pv = G.nodes[v].get("pos_m") or G.nodes[v].get("location_m")
        if pu is None or pv is None:
            continue
        cv2.line(bg, tuple(to_px(pu)), tuple(to_px(pv)), (200, 200, 0), 1)
    for n, d in G.nodes(data=True):
        p = d.get("pos_m") or d.get("location_m")
        if p is None:
            continue
        color = (0, 100, 255) if d["type"] == "entry_exit" else (0, 200, 255)
        cv2.circle(bg, tuple(to_px(p)), 8, color, -1)
        cv2.putText(bg, f'{n}:{d["type"]}', tuple(to_px(p) + [10, 4]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    viz = f"outputs/graph_{scene_id}_static.png"
    cv2.imwrite(viz, bg)

    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    for n, d in G.nodes(data=True):
        print(f'  {n}: {d["type"]}')
    print(f"Saved: {out} and {viz}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    main(ap.parse_args().scene)