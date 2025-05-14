import torch
from datetime import datetime
import os
from ultralytics import YOLO

if __name__ == "__main__":
    # TOGGLE PER RUN
    mode = "train"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_folder = "data/yolo_dataset/images/train/"
    run_name = f"{mode}_yolorun_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    torch.cuda.empty_cache()

    if mode == "pretrained":
        model = YOLO("yolo11n-obb.pt")
        results = model.predict([data_folder + "0_1688827660979_frame0.jpg"],
                                imgsz=(1080, 1920),
                                device=device,
                                project="/home/gridsan/wyang/super_urop_workspace/whale_vision/logs",
                                name=run_name,
                                save=True,
                                )
    elif mode == "train":
        model = YOLO("yolo11n-obb.pt")
        results = model.train(data="data/yolo_dataset.yaml", 
                            epochs=150, 
                            imgsz=1280, 
                            device=device, 
                            pretrained=True, 
                            rect=False,
                            project="/home/gridsan/wyang/super_urop_workspace/whale_vision/logs",
                            name=run_name,
                            multi_scale=False,
                            overlap_mask=False,
                            batch=16,
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
                            copy_paste=0,
                            erasing=0.0)

    elif mode == "val":
        model_weights = "logs/train_yolorun_2025-01-29_09-28-21/weights/best.pt"
        model = YOLO(model_weights)
        metrics = model.val(batch=8)
        print(metrics)

    else:     
        # get latest model run
        model_weights = "logs/train_yolorun_2025-01-29_09-28-21/weights/last.pt" 
        model = YOLO(model_weights)
        with torch.no_grad():
            images = os.listdir(data_folder)
            results = model.predict(source=[data_folder + images[i] for i in range(20)],
                                    conf=0.25,
                                    imgsz = 640,
                                    device=device,
                                    project="/home/gridsan/wyang/super_urop_workspace/whale_vision/logs",
                                    name=run_name,
                                    show=True,
                                    save_txt=True,
                                    show_boxes=True,
                                    show_labels=True,
                                    show_conf=True,
                                    save=True
                                    )
            for r in results:
                for b in r["boxes"]:
                    print(b.xyzy.tolist())