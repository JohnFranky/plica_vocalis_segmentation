import os
import cv2
import re
import numpy as np
import scipy.io


def loadImages(path):
    images = list()
    number_files = len(os.listdir(path))

    for i in range(1, number_files):
        #x = path + '{0:05d}'.format(i) + ".png"
        images.append(cv2.imread(path + '{0:05d}'.format(i) + ".png", 0))

    return images

def createBlackLines(img):
    blackImg = img.copy()

    for x in range(192, 198):
        blackImg[266, x] = 0
    blackImg[267, 192] = 0
    blackImg[267, 191] = 0
    blackImg[268, 191] = 0
    blackImg[268, 190] = 0
    blackImg[269, 190] = 0
    blackImg[269, 189] = 0
    blackImg[270, 189] = 0

    for x in range(178, 190):
        blackImg[270, x] = 0
    y = 278
    for x in range(168, 178):
        blackImg[y, x] = 0
        blackImg[y+1, x] = 0
        y -= 1
    for x in range(278, 298):
        blackImg[x, 168] = 0

    return blackImg




if __name__ == "__main__":
    Initials = "RH"   
    if(Initials == "RH"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\RH"
        image_path = path + r"\png\\"
        snake_path = path + r"\Segmentation\3_14500-15000\[1-475]" +r"\3_14500-15000[1-475].snake"
        laserdots_path = path + r"\results\laserdots\475.mat"

    images = loadImages(image_path)
    FrameNumberInt = 1
    for image in images:
        bo = createBlackLines(image)
        #cv2.imshow("SnakeInfo",bo)
        #cv2.waitKey(0)
        save_path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\RH\Segmentation\blackoutlineImages"
        if(FrameNumberInt < 10):
            FrameNumber = r"000"+str(FrameNumberInt)
        elif(FrameNumberInt < 100):
            FrameNumber = r"00"+str(FrameNumberInt)
        elif(FrameNumberInt < 1000):
            FrameNumber = r"0"+str(FrameNumberInt)
        else:
            FrameNumber = str(FrameNumberInt)
        save_path = save_path + r"\0" + FrameNumber + r".png"
        result=cv2.imwrite(save_path, bo)
        FrameNumberInt += 1