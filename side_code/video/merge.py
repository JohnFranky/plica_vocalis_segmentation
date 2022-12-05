import cv2
import numpy as np
import os

path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\videos\2_cropped"
path_goal = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\videos\2_merge"
pic_path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\MS\png"
imgs = os.listdir(path)
imgsOri = os.listdir(pic_path)

c = 0
for i in imgs:
    img = cv2.imread(os.path.join(path,i))
    imgOri = cv2.imread(os.path.join(pic_path,imgsOri[c]))
    x = imgOri | img
    cv2.imwrite(os.path.join(path_goal,i), x)
    c += 1
