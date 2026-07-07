"""Open-vocabulary static object detection on the clean background frame.

Grounding DINO is text-conditioned: the prompt must be lowercase,
period-separated ("door. car."). Runs once per scene.

Vocabulary priority: VLM-proposed scene vocabulary (scene_vocab_{scene}.json)
if present, else per-scene config, else defaults.

Post-processing:
  1) drop detections whose label collapsed to a token fragment
  2) cross-class NMS (IoU > 0.5 -> keep higher confidence)
  3) same-class containment filter (IoS > 0.8 -> drop the smaller box);
     catches full-object + fragment duplicates that IoU-NMS misses

Usage:
    python src/perception/objects.py zara01
    python src/perception/objects.py zara01 --image path/to/other.png  (debug)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
import yaml
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONFIG = Path("configs/scenes.yaml")
MODEL_ID = "IDEA-Research/grounding-dino-base"
DEFAULT_LABELS = ["storefront window", "door", "car", "tree", "bench", "trash bin"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_IOU = 0.5
CONTAIN_IOS = 0.8


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0


def ios(a, b):
    """Intersection over the smaller box: ~1.0 when one box sits inside
    the other, even if IoU is low. Catches object + its-own-part duplicates."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return inter / smaller if smaller > 0 else 0


def main(scene_id: str, image_override: str | None):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[scene_id]
    vocab_file = Path(f"data/processed/scene_vocab_{scene_id}.json")
    if vocab_file.exists():
        labels = json.loads(vocab_file.read_text(encoding="utf-8"))
        print(f"Using VLM scene vocabulary: {labels}")
    else:
        labels = cfg.get("object_prompt", DEFAULT_LABELS)
    text = " ".join(f"{l.lower()}." for l in labels)

    img_path = Path(image_override) if image_override \
        else Path(f"data/processed/background_{scene_id}.png")
    assert img_path.exists(), f"Image not found: {img_path}"
    image = Image.open(img_path).convert("RGB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    names = res.get("text_labels", res["labels"])
    objects = []
    for i, (box, score, label) in enumerate(zip(res["boxes"], res["scores"], names)):
        objects.append(dict(
            id=f"object_{i:02d}",
            type=str(label),
            bbox=[round(float(v), 1) for v in box.tolist()],
            confidence=round(float(score), 3),
            source="grounding-dino-base",
        ))

    # 1) Drop label fragments ("a", "a tr")
    valid = {l.lower() for l in labels}
    objects = [o for o in objects if o["type"].strip() in valid]

    # 2) Cross-class NMS: overlapping duplicates, keep higher confidence
    objects.sort(key=lambda o: -o["confidence"])
    kept = []
    for o in objects:
        if all(iou(o["bbox"], k["bbox"]) < NMS_IOU for k in kept):
            kept.append(o)
    objects = kept

    # 3) Same-class containment: a fragment box inside a full-object box
    #    (car + its hood). Big boxes first; a smaller box survives only if
    #    it is not sitting inside an already-kept box of the same class.
    objects.sort(key=lambda o: -(o["bbox"][2] - o["bbox"][0])
                                * (o["bbox"][3] - o["bbox"][1]))
    kept = []
    for o in objects:
        contained = any(o["type"] == k["type"]
                        and ios(o["bbox"], k["bbox"]) > CONTAIN_IOS
                        for k in kept)
        if not contained:
            kept.append(o)
    objects = kept

    suffix = "_debug" if image_override else ""
    out = Path(f"data/processed/objects_{scene_id}{suffix}.json")
    out.write_text(json.dumps(objects, indent=2), encoding="utf-8")

    img = cv2.imread(str(img_path))
    for o in objects:
        x1, y1, x2, y2 = map(int, o["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(img, f'{o["type"]} {o["confidence"]:.2f}', (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    viz = Path(f"outputs/objects_{scene_id}{suffix}.png")
    cv2.imwrite(str(viz), img)

    print(f"Prompt: {text}")
    print(f"Objects found: {len(objects)}")
    for o in objects:
        print(f'  {o["id"]}: {o["type"]} ({o["confidence"]})')
    print(f"Saved: {out} and {viz}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--image", default=None, help="override input image (debug)")
    args = ap.parse_args()
    main(args.scene, args.image)