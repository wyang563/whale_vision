# debugging playground

from ultralytics import YOLO
from PIL import Image

model = YOLO("yolo11n-obb.pt")
test_image = Image.open("data/yolo_dataset/images/train/0_1688829151574_frame0.jpg")
results = model.predict([test_image], imgsz=(1080, 1920), device=0, project="logs", name="debug", rect=True, save=True)
print(results)
