"""VLM proposes the scene object vocabulary from the closed ontology.

Looks at the clean background frame and answers: which of the allowed
object types are present in this scene? Output feeds Grounding DINO.

Usage: python src/perception/scene_vocab.py zara01
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ONTOLOGY = Path("configs/ontology.yaml")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def main(scene_id: str):
    types = yaml.safe_load(ONTOLOGY.read_text(encoding="utf-8"))["object_types"]
    bg = Path(f"data/processed/background_{scene_id}.png").resolve()
    assert bg.exists(), f"Run background.py for {scene_id} first"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    prompt = (
        "This is a static background of a fixed-camera scene (no people). "
        "Task: identify which object types from the allowed list are clearly "
        "visible in the image.\n"
        "Rules:\n"
        "- Use ONLY strings from the allowed list, verbatim. Do not invent types.\n"
        "- Include a type only if you clearly see it.\n"
        "- Check the full frame: buildings and their features, street "
        "furniture, vegetation, vehicles.\n"
        "- Return a JSON array of strings. No other text.\n"
        f"Allowed list: {json.dumps(types)}"
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{bg}"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs,
                       return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    reply = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Parse and validate against the ontology (closed vocabulary!)
    try:
        proposed = json.loads(reply.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        sys.exit(f"VLM returned non-JSON: {reply}")
    vocab = [t for t in proposed if t in types]
    dropped = [t for t in proposed if t not in types]

    out_path = Path(f"data/processed/scene_vocab_{scene_id}.json")
    out_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    print(f"VLM raw reply: {reply}")
    print(f"Scene vocabulary: {vocab}")
    if dropped:
        print(f"Dropped (not in ontology): {dropped}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    main(ap.parse_args().scene)