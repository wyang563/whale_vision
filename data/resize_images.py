import cv2
import os
from tqdm import tqdm
import csv

# resize images to size 1920 x 1920
mode = "val"

old_dataset = f"data/rcnn_segment_dataset/images/{mode}/"
new_dataset = f"data/yolo_dataset/images/{mode}/"

old_dataset_labels = f"data/rcnn_segment_dataset/{mode}_segmented_boxes.csv"
new_dataset_labels = f"data/yolo_dataset/labels/{mode}/"

with open(f"data/rcnn_segment_dataset/{mode}_segmented_boxes.csv", "r") as f:
    reader = csv.reader(f)
    boxes = list(reader)

for img in tqdm(os.listdir(old_dataset)):
    image = cv2.imread(old_dataset + img)
    resized_image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
    padded_image = cv2.copyMakeBorder(resized_image, 420, 420, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.imwrite(new_dataset + img, padded_image)
    img_number = img.split("_")[0]
    img_file_prefix = img.split(".")[0]
    box_data = boxes[int(img_number)]

    with open(new_dataset_labels + f"{img_file_prefix}.txt", "w") as f:
        for j in range(1, len(box_data), 8):
            if "nan" not in box_data[j:j+8]:
                out_row = [float(x) for x in box_data[j:j+8]]
                for k in range(0, len(out_row), 2):
                    out_row[k] /= 2560
                    out_row[k + 1] = (out_row[k + 1] + 560) / 2560

                out_row = [str(x) for x in out_row]
                f.write("0 " + " ".join(out_row) + "\n")



