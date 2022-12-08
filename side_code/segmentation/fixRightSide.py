import cv2
import os

if __name__ == "__main__":
    Initials = "DD"
    FrameNumberStart = 1
    FrameNumberEnd = 350 #number of last frame included

    path = r"E:\_\HLE_Dataset\DD"


    result_path = r"\Segmentation" + r"\plica_vocalis"
    image_path = path + r"\Segmentation\plica_vocalis"
    for k in range (FrameNumberStart, FrameNumberEnd + 1):
        FrameNumberInt = k
        if(FrameNumberInt < 10):
            FrameNumber = r"0000"+str(FrameNumberInt)
        elif(FrameNumberInt < 100):
            FrameNumber = r"000"+str(FrameNumberInt)
        elif(FrameNumberInt < 1000):
            FrameNumber = r"00"+str(FrameNumberInt)
        else:
            FrameNumber = r"0"+str(FrameNumberInt)

        number = FrameNumber + r".png"
        first_img_path = result_path + number
        first_img_path = os.path.join(path, "Segmentation", "plica_vocalis", number)
        
        #Loading images
        first_image = cv2.imread(first_img_path, 0)
        cv2.imshow('Original', first_image)
        binaryImg = first_image.copy()

        for x in range(0, 3):
            for y in range(0, 512):
                binaryImg[y, 255 - x] = 0.0

        cv2.imshow("2", binaryImg)
        #cv2.waitKey(0)
        cv2.imwrite(first_img_path, binaryImg)
    