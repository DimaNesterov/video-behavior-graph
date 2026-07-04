from ultralytics import YOLO
import torch

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
model = YOLO("yolo11n.pt")  # скачает ~5 МБ веса при первом запуске
model.to("cuda")
print("YOLO on:", model.device)