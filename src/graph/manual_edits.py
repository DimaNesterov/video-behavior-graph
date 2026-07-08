"""Manual graph edits: user-added/suppressed objects, one overlay per scene.

This is the backend of the future editing UI (phase 4): the UI will call
these functions; until then they are usable from scripts and the CLI.

Usage (CLI):
    python src/graph/manual_edits.py zara01 add "shop window" 0 11 460 246
    python src/graph/manual_edits.py zara01 list
    python src/graph/manual_edits.py zara01 remove manual_00
"""
import argparse
import json
from pathlib import Path

MANUAL_DIR = Path("data/manual")


def _path(scene_id: str) -> Path:
    return MANUAL_DIR / f"{scene_id}_objects.json"


def load(scene_id: str) -> list:
    p = _path(scene_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def save(scene_id: str, objects: list) -> None:
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)   # folder auto-created
    _path(scene_id).write_text(json.dumps(objects, indent=2), encoding="utf-8")


def add_object(scene_id: str, obj_type: str, bbox: list[float]) -> dict:
    objects = load(scene_id)
    ids = {o["id"] for o in objects}
    n = 0
    while f"manual_{n:02d}" in ids:
        n += 1
    obj = dict(id=f"manual_{n:02d}", type=obj_type,
               bbox=[float(v) for v in bbox],
               confidence=1.0, source="manual")
    objects.append(obj)
    save(scene_id, objects)
    return obj


def remove_object(scene_id: str, obj_id: str) -> bool:
    objects = load(scene_id)
    kept = [o for o in objects if o["id"] != obj_id]
    save(scene_id, kept)
    return len(kept) < len(objects)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("command", choices=["add", "list", "remove"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.command == "add":
        obj = add_object(a.scene, a.args[0], a.args[1:5])
        print(f"Added: {obj}")
    elif a.command == "list":
        for o in load(a.scene):
            print(o)
    elif a.command == "remove":
        ok = remove_object(a.scene, a.args[0])
        print("Removed" if ok else "Not found")