# script to convert current rcnn dataset to desired yolo format

import csv
import os
from tqdm import tqdm

def find_file_name(number, file_list):
    for f in file_list:
        if f.startswith(f"{number}_"):
            return f.split(".")[0] + ".txt"

with open("data/rcnn_segment_dataset/train_segmented_boxes.csv", "r") as f:
    reader = csv.reader(f)
    boxes = list(reader)

file_list = [f for f in os.listdir("data/rcnn_segment_dataset/images/train") if f != ".DS_Store"]

class_num = 0
for i, row in tqdm(enumerate(boxes)):
    for j in range(1, len(row), 8):
        if "nan" not in row[j:j+8]:
            out_file = find_file_name(i, file_list)

            # convert to int strings
            out_row = [float(x) for x in row[j:j+8]]

            # normalize pixel coordinates between 0 and 1
            for k in range(0, len(out_row), 2):
                out_row[k] /= 3840
                out_row[k + 1] /= 2560

            out_row = [str(x) for x in out_row]
            with open("data/rcnn_segment_dataset/labels/train/" + out_file, "a") as f:
                f.write(str(class_num) + " " + " ".join(out_row) + "\n")

            
