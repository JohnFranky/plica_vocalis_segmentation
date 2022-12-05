import cv2
import numpy as np
import os

path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\videos"
imgs = os.listdir(os.path.join(path, "2"))


for i in imgs:
    img = cv2.imread(os.path.join(path, "2",i))
    for j in range(0,4):
        x = img[0:img.shape[0]-4, (0 + j*int(img.shape[1]/4)):(256+j*int(img.shape[1]/4))]
        #cv2.imshow("cropped", x)
        #cv2.waitKey(0)
        if(len(i) >= 11):
             cv2.imwrite(os.path.join(path, "2_cropped",i[0:7]+ "_" +str(j)+i[7:12]), x)
        else:
            cv2.imwrite(os.path.join(path, "2_cropped",i[0:6]+"_"+str(j)+i[6:11]), x)
