"""VLM verification of DINO boxes: crop each box, ask yes/no.

Reads objects_{scene}.json, writes objects_{scene}_verified.json.
Rejected boxes are kept in a separate file for inspection.

Scene-specific footage context (e.g. "blurry 2009 CCTV") comes from
configs/scenes.yaml (footage_note); the code itself is scene-agnostic.

Usage: python src/perception/verify_objects.py zara01
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONFIG = Path("configs/scenes.yaml")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PAD = 60  # px of context around each crop


def main(scene_id: str):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]
    footage = cfg.get("footage_note", "")
    context = f"This is a crop from {footage}. " if footage else ""

    obj_path = Path(f"data/processed/objects_{scene_id}.json")
    objects = json.loads(obj_path.read_text(encoding="utf-8"))
    bg = Image.open(f"data/processed/background_{scene_id}.png").convert("RGB")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    crops_dir = Path("data/processed/crops")
    crops_dir.mkdir(exist_ok=True)

    accepted, rejected = [], []
    EDGE_PX = 5
    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        at_edge = (x1 <= EDGE_PX or y1 <= EDGE_PX
                   or x2 >= bg.width - EDGE_PX or y2 >= bg.height - EDGE_PX)
    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        crop = bg.crop((max(0, x1 - PAD), max(0, y1 - PAD),
                        min(bg.width, x2 + PAD), min(bg.height, y2 + PAD)))
        crop_path = crops_dir / f'{scene_id}_{o["id"]}.png'
        crop.save(crop_path)

        partial = ("The object is cut off by the frame edge, so only part "
                   "of it is visible. " if at_edge else "")
        prompt = (f'{context}{partial}'
                  f'Could the main object in this image plausibly be a {o["type"]}? '
                  'Answer with exactly one word: yes or no.')
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{crop_path.resolve()}"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs,
                           return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        reply = processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True)[0].strip().lower()

        o["vlm_verdict"] = reply
        (accepted if reply.startswith("yes") else rejected).append(o)
        print(f'{o["id"]} {o["type"]} ({o["confidence"]}): {reply}')

    Path(f"data/processed/objects_{scene_id}_verified.json").write_text(
        json.dumps(accepted, indent=2), encoding="utf-8")
    Path(f"data/processed/objects_{scene_id}_rejected.json").write_text(
        json.dumps(rejected, indent=2), encoding="utf-8")
    print(f"\nAccepted: {len(accepted)} | Rejected: {len(rejected)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    main(ap.parse_args().scene)