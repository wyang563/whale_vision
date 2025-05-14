#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH -c 4



python scripts/inference.py --file_path data/yolo_dataset/images/val/83944.jpg --model logs/train_yolorun_2025-02-18_04-22-37/weights/last.pt --conf_threshold 0.1 --img_size 640 --project logs