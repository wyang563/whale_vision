import cv2
import numpy as np

frame = cv2.imread("data/yolo_dataset/images/train/0_1688829151574_frame0.jpg")
print(frame.shape)
# frame = cv2.resize(frame, (2560, 1440))

full_bound_box = "0 0.463671875 0.6296875 0.5578125 0.56484375 0.574609375 0.589453125 0.48046875 0.654296875"
full_bound_box = [int(1920 * float(x)) for x in full_bound_box.split()[1:] if x != "nan"]
print(full_bound_box)

# full_bound_box = "0,nan,nan,nan,nan,nan,nan,nan,nan,1187.0,1052.0,1428.0,886.0,1471.0,949.0,1230.0,1115.0,1453.0,875.0,1463.0,832.0,1646.0,875.0,1635.0,918.0,1339.0,969.0,1450.0,969.0,1450.0,1219.0,1339.0,1219.0,1074.0,910.0,1342.0,844.0,1357.0,902.0,1088.0,968.0,1481.0,900.0,1559.0,893.0,1573.0,1053.0,1495.0,1060.0,742.0,834.0,1081.0,828.0,1083.0,915.0,744.0,921.0,1241.0,861.0,1458.0,861.0,1458.0,887.0,1241.0,887.0,1447.0,793.0,1675.0,793.0,1675.0,819.0,1447.0,819.0,1173.0,793.0,1267.0,793.0,1267.0,828.0,1173.0,828.0,nan,nan,nan,nan,nan,nan,nan,nan"
# full_bound_box = [float(x) for x in full_bound_box.split(",")[1:] if x != "nan"]

dot_radius = 5
dot_thickness = -1
dot_color = (0, 255, 0)  # Green
center_color = (255, 0, 0)

for i in range(0, len(full_bound_box), 8):
    for j in range(i, i + 8, 2):
        dot_x, dot_y = round(full_bound_box[j]), round(full_bound_box[j + 1])
        cv2.circle(frame, 
                (dot_x, dot_y),
                dot_radius, 
                dot_color, 
                dot_thickness)
    
cv2.imwrite("frame_with_dots.jpg", frame) 