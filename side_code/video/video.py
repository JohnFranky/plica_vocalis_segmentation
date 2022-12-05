import cv2
import numpy as np
import os

Initials = r"MS"
art = "points"
path = os.path.join(r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\videos\MS_Y")#\heatmaps",art,"heatmaps",Initials)
imgs = sorted(os.listdir(path))
img_array = []

for i in imgs:
    img = cv2.imread(os.path.join(path, i))
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


#out = cv2.VideoWriter(Initials+art+'merged.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
out = cv2.VideoWriter('MS_GT.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()