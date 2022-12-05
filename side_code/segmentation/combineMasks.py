import os
import cv2
import re
import numpy as np
import scipy.io

def recolorImage(img, goal):
    imgCopy = img.copy()
    for x in range(0,len(img)):
        for y in range(0,len(img[0])):
            if(not(img[x,y] == 0)):
                imgCopy[x,y] = goal
    return imgCopy

def combine(glottis, vocalis, laser, colorGlottis, colorVocalis, colorLaser):
    imgCopy = glottis.copy()
    for x in range(0,len(imgCopy)):
        for y in range(0,len(imgCopy[0])):
            if(glottis[x,y] == 255):
                imgCopy[x,y] = colorGlottis
            elif(laser[x,y] == 255):
                imgCopy[x,y] = colorLaser
            elif(vocalis[x,y] == 255):
                imgCopy[x,y] = colorVocalis
            else:
                continue
    return imgCopy


if __name__ == "__main__":
    m = 9   
    while m < 10:
        anzFrames = 1
        if(m == 0):
            path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\CF"
            anzFrames = 329
            Initials = "CF"
        elif(m == 1):
            path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\CM"
            Initials = "CM"
            anzFrames = 311
        elif(m == 2):
            path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\DD"
            Initials = "DD"
            anzFrames = 350
        elif(m == 3):
            path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\FH"
            Initials = "FH"
            anzFrames = 100
        elif(m == 4):
            path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\LS"
            Initials = "LS"
            anzFrames = 350
        elif(m == 5):
            path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\MK"
            Initials = "MK"
            anzFrames = 474
        elif(m == 6):
            path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\MS"
            Initials = "MS"
            anzFrames = 251
        elif(m == 7):
            path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\RH"
            Initials = "RH"
            anzFrames = 327
        elif(m == 8):
            path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\SS"
            Initials = "SS"
            anzFrames = 225
        elif(m == 9):
            path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\TM"
            Initials = "TM"
            anzFrames = 603
        else:
            path = "error"
        """
        test = cv2.imread(os.path.join(path, "Segmentation", "all_masks","00001CF.png"),0)
        cv2.imshow("test",test)
        cv2.waitKey(0)
        """
            
        test2 = cv2.imread(os.path.join(path, "png","00001TM.png"),0)
        glottis_path = os.path.join(path, "Segmentation", "glottis")# + Initials + ".png")
        vocalis_path = os.path.join(path, "Segmentation", "plica_vocalis")
        laser_path = os.path.join(path, "Segmentation", "laserdots")
        

        FrameNumberInt = 0
        for i in range(0,anzFrames):
            FrameNumberInt += 1
            if(FrameNumberInt < 10):
                FrameNumber = r"0000"+str(FrameNumberInt)+ Initials + ".png"
            elif(FrameNumberInt < 100):
                FrameNumber = r"000"+str(FrameNumberInt)+ Initials + ".png"
            elif(FrameNumberInt < 1000):
                FrameNumber = r"00"+str(FrameNumberInt)+ Initials + ".png"
            else:
                FrameNumber = r"0"+str(FrameNumberInt)+ Initials + ".png"
            glottisPath = os.path.join(glottis_path,FrameNumber)
            vocalisPath = os.path.join(vocalis_path,FrameNumber)
            laserPath = os.path.join(laser_path,FrameNumber)

            glottis = cv2.imread(glottisPath,0)
            vocalis = cv2.imread(vocalisPath,0)
            laser = cv2.imread(laserPath,0)

            cv2.imshow("test2",test2|laser.astype(np.uint8))
            cv2.imwrite(os.path.join(path, "Segmentation", "image.png"), test2|laser.astype(np.uint8))
            cv2.waitKey(0)

            """
            cv2.imshow("glottis", glottis)
            cv2.imshow("vocalis", vocalis)
            cv2.imshow("laser", laser)
            cv2.waitKey(0)
            glottis = recolorImage(vocalis, 60)
            vocalis = recolorImage(vocalis, 150)
            laser = recolorImage(laser, 210)
            """
            colorGlottis = 1#255 #1
            colorVocalis = 2#85 #2
            colorLaser = 3#170 #3
            erg = combine(glottis, vocalis, laser, colorGlottis, colorVocalis, colorLaser)
            """
            cv2.imshow("erg", erg)
            cv2.waitKey(0)
            """
            print("Saving image " + FrameNumber)
            result_path = os.path.join(path, "Segmentation", "all_masks", FrameNumber)
            cv2.imwrite(result_path, erg)
        m += 1