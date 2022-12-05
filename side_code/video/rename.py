import cv2
import numpy as np
import os
c = 0
while(c<6):
    if(c == 0):
        Initials = r"MK"
        art = "stimmlippe"
    elif(c== 1):
        Initials = r"MK"
        art = "glottis"
    elif(c== 2):
        Initials = r"MK"
        art = "points"
    elif(c == 3):
        Initials = r"MS"
        art = "stimmlippe"
    elif(c== 4):
        Initials = r"MS"
        art = "glottis"
    else:
        Initials = r"MS"
        art = "points"
    path = os.path.join(r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\videos\MK_P\saved_images")#\heatmaps",art,"heatmaps",Initials)
    imgs = sorted(os.listdir(path))
    img_array = []

    for i in imgs:
        img = cv2.imread(os.path.join(path, i))
        x = i[0:5]
        rest = i[5:]
        png = rest[-4:]
        lenge = len(rest)-len(png)
        if(lenge == 1):
            rest = "000"+rest
        elif(lenge == 2):
            rest = "00"+rest
        elif(lenge == 3):
            rest = "0"+rest
        cv2.imwrite(os.path.join(path,(x+rest)), img)
        os.remove(os.path.join(path, i))
        

    c+=1