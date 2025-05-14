from ultralytics import RTDETR, YOLO
import torch
from datetime import datetime
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

if __name__ == "__main__":
    # parameters (toggle before each run)
    # model types: rtdetr-l, yolo11n-obb, yolov8n-obb
    model_type = "yolov8s-obb"
    model_path = "whale_vision/logs/train_yolov8s-obb.pt_2025-05-14_12-49-27/weights/best.pt"
    data_yaml = "whale_vision/data/yolo_dataset_m.yaml"
    mode = "val"

    if "yolo11" in model_type:
        model = YOLO(model_path)
    elif "yolov8" in model_type:
        model = YOLO(model_path)
    else:
        model = RTDETR(model_path)

    run_name = f"{mode}_{os.path.basename(model_type)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    metrics = model.val(data=data_yaml, 
                        batch=16,
                        imgsz=640,
                        conf=0.3,
                        device=device,
                        project="/home/alex/multienv_sim/whale_vision/logs",
                        name=run_name,
                        verbose=False,
                        plots=True)

