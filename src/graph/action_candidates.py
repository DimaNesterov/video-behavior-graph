"""Rule-based action hypotheses over the dynamic graph (spec section 12).

Closed action vocabulary; every hypothesis is a deterministic rule over
one graph snapshot (plus a short history for enter/exit):

  continue          -- moving, no strong relation to anything
  stop_near_object  -- stopped within NEAR of an object
  approach_object   -- moving toward a nearby object (near + approaching)
  head_to_exit      -- heading_to edge to a boundary exit
  enter_building    -- heading_to edge to a building entry, close by
  wander            -- moving slowly with no alignment to anything

Output: per-timestep action labels per person, appended into the dynamic
graph JSON as 'action_candidates'.

Usage: python src/graph/action_candidates.py zara01
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

WANDER_SPEED = 0.8      # slow but not stopped
STOP_OBJ_M = 3.0        # stopped near an object counts within this range
ENTER_M = 4.0           # enter_building requires being this close


def main(scene_id: str, source: str):
    path = Path(f"data/processed/graph_{scene_id}_dynamic_{source}.json")
    frames = json.loads(path.read_text(encoding="utf-8"))

    static = json.loads(Path(
        f"data/processed/graph_{scene_id}_static.json").read_text(encoding="utf-8"))
    entry_subtype = {n["id"]: n.get("subtype", "boundary_exit")
                     for n in static["nodes"] if n["type"] == "entry_exit"}

    counts = defaultdict(int)
    for fr in frames:
        edges_by_person = defaultdict(list)
        for e in fr["edges"]:
            edges_by_person[e["source"]].append(e)

        actions = []
        for p in fr["persons"]:
            pe = edges_by_person[p["id"]]
            near = [e for e in pe if e["type"] == "near"]
            heading = [e for e in pe if e["type"] == "heading_to"]

            label, target = "continue", None
            if p["stopped"]:
                close = [e for e in near if e["distance_m"] < STOP_OBJ_M]
                if close:
                    tgt = min(close, key=lambda e: e["distance_m"])
                    label, target = "stop_near_object", tgt["target"]
                else:
                    label = "stop"
            else:
                entering = [e for e in heading
                            if entry_subtype.get(e["target"]) == "building_entry"
                            and e["distance_m"] < ENTER_M]
                approaching = [e for e in near if e.get("approaching")]
                exiting = [e for e in heading
                           if entry_subtype.get(e["target"]) == "boundary_exit"]
                if entering:
                    tgt = min(entering, key=lambda e: e["distance_m"])
                    label, target = "enter_building", tgt["target"]
                elif approaching:
                    tgt = min(approaching, key=lambda e: e["distance_m"])
                    label, target = "approach_object", tgt["target"]
                elif exiting:
                    tgt = min(exiting, key=lambda e: e["distance_m"])
                    label, target = "head_to_exit", tgt["target"]
                elif p["speed"] < WANDER_SPEED:
                    label = "wander"

            actions.append(dict(person=p["id"], action=label, target=target))
            counts[label] += 1
        fr["action_candidates"] = actions

    path.write_text(json.dumps(frames), encoding="utf-8")
    total = sum(counts.values())
    print(f"Labeled {total} person-timesteps:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>18}: {v:>6} ({v / total:.1%})")
    print(f"Updated: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--source", default="gt", choices=["gt", "tracked"])
    args = ap.parse_args()
    main(args.scene, args.source)