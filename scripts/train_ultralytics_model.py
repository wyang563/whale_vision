from ultralytics import RTDETR, YOLO
import torch
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # parameters (toggle before each run)
    # model types: rtdetr-l, yolo11n-obb, yolov8n-obb
    model_type = "yolov8s-obb.pt"
    data_yaml = "whale_vision/data/yolo_dataset.yaml"
    mode = "train"

    if "rtdetr" in model_type:
        model = RTDETR(model_type)
        batch_size = 4  # Smaller batch size for RTDETR
    elif "yolo11" in model_type:
        # Initialize YOLO11 model with OBB task
        model = YOLO(model_type, task='obb')
        batch_size = 16
    elif "yolov8" in model_type:
        # Initialize YOLO model with OBB task
        model = YOLO(model_type, task='obb')
        batch_size = 16

    run_name = f"{mode}_{model_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Train the model on the COCO8 example dataset for 100 epochs
    if "yolo11" in model_type or "yolov8" in model_type:
        results = model.train(data=data_yaml, 
                          epochs=100, 
                          imgsz=1280, 
                          pretrained=True,
                          device=device,
                          project="/home/alex/multienv_sim/whale_vision/logs",
                          name=run_name,
                          overlap_mask=False,
                          batch=batch_size,
                          hsv_h=0.03,
                          hsv_s=0.8,
                          hsv_v=0.6,
                          degrees=180,
                          translate=0.3,
                          scale=0.75,
                          shear=0,
                          flipud=0.5,
                          fliplr=0.5,
                          mosaic=0,
                          mixup=0,
                          erasing=0,
                          copy_paste=0)
    else:
        results = model.train(data=data_yaml, 
                          epochs=100, 
                          imgsz=1280, 
                          pretrained=True,
                          device=device,
                          project="/home/alex/multienv_sim/whale_vision/logs",
                          name=run_name,
                          overlap_mask=False,
                          batch=batch_size,
                          hsv_h=0.03,
                          hsv_s=0.8,
                          hsv_v=0.6,
                          degrees=180,
                          translate=0.3,
                          scale=0.75,
                          shear=0,
                          flipud=0.5,
                          fliplr=0.5,
                          mosaic=0,
                          mixup=0,
                          copy_paste=0)
