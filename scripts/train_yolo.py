import torch
import torch.nn as nn
from tqdm import tqdm
from configs.load_config import load_config
from datetime import datetime
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

config = load_config("configs/config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

# logger setup code
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

train_run_name = f"yolorun_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


model = YOLO("yolo11n-obb.pt")
results = model.train(data="data/rcnn_segment_dataset.yaml", 
                    epochs=10, 
                    imgsz=[2560, 1440], 
                    device=0, 
                    pretrained=True, 
                    rect=True,
                    project="/home/gridsan/wyang/super_urop_workspace/whale_vision/logs",
                    name=train_run_name)
