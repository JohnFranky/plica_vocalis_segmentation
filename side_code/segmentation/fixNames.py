import cv2
import os

if __name__ == "__main__":
    Initials = "CF"
    FrameNumberStart = 1
    FrameNumberEnd = 701 #number of last frame included

    path = r"E:\_\HLE_Dataset\CF"


    image_path = path + r"\png"
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
        first_img_path = os.path.join(path, "png", number)
        number = FrameNumber + r"CF.png"
        result_path = os.path.join(path, "png", number)
        #Loading images
        first_image = cv2.imread(first_img_path, 0)

        cv2.imwrite(result_path, first_image)
        os.remove(first_img_path)