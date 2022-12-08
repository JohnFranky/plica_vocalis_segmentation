import re
import cv2
import os
from operator import itemgetter
import numpy as np

def getSnakesAtFrame(path, frame_number):
    snakefile = open(path, mode="r")
    lines = snakefile.readlines()

    currentFrame = -1
    goal = False
    snakes = []

    for line in lines:
        if re.fullmatch("^[0-9][0-9][0-9]\n$", line) or re.fullmatch("^[0-9][0-9]\n$", line) or re.fullmatch("^[0-9]\n$", line):
            currentFrame += 1 
            if(currentFrame == frame_number):
                goal = True
            elif (currentFrame > frame_number):
                break

        # MATLAB is one initialized
        if(goal):
            if re.fullmatch("^.*\t.*\n$", line):
                a, b = line.replace("\t", " ").replace("\n", "").replace(".00", "").split(" ")
                snakes.append([int(a), int(b)])
    return snakes

def drawGlottis(binaryImg, snakes, min_y_id, max_y_id, min_y, max_y):
    snakes = sorted(snakes , key=lambda k: [k[1], k[0]])
    if(snakes == []):
        return binaryImg
    resnake = []
    tmp = []
    y = min_y[1]
    for snake in snakes:
        if(snake[1] == y):
            tmp.append(snake)
        else:
            resnake.append(tmp)
            y += 1
            while y < snake[1]: #temporal fix: approximation of undefined lines
                tmp1 = []
                for sn in tmp:
                    tmp1.append([sn[0], sn[1]+1])
                resnake.append(tmp1)
                y+=1
            tmp = []
            tmp.append(snake)

      
    for snake in resnake:
        if(len(snake) == 0):
            continue
        else:
            for i in range(len(snake)):
                if(i == len(snake)-1):
                    binaryImg[snake[i][1], snake[i][0]] = 255
                    break
                else:
                    x0 = snake[i][0]
                    x1 = snake[i+1][0]
                    binaryImg[snake[i][1], snake[i][0]] = 255
                    for diff in range(x0, x1):
                        binaryImg[snake[i][1], diff] = 255
        #cv2.imshow("Line", binaryImg)
        #cv2.waitKey(0)  
    return binaryImg


if __name__ == "__main__":
    Initials = "TM"
    FrameNumberStart = 1
    FrameNumberEnd = 605 #number of last frame included
    if(Initials == "CF"):
        path = r"_\HLE_Dataset\CF"
        snake_path = path + r"\Segmentation\2_9500_10200\[1-701]" +r"\2_9500_10200[1-701].snake"
    elif(Initials == "CM"):
        path = r"_\HLE_Dataset\CM"
        snake_path = path + r"\Segmentation\1_200-700\[190-501]" +r"\1_200-700[190-501].snake"
    elif(Initials == "DD"):
        path = r"_\HLE_Dataset\DD"
        snake_path = path + r"\Segmentation\Denis2_15518-16018\[1-350]" +r"\Denis2_15518-16018[1-350].snake"
    elif(Initials == "FH"):
        path = r"_\HLE_Dataset\FH"
        snake_path = path + r"\Segmentation\4_2000-2650\[1-651]" +r"\4_2000-2650[1-651].snake"
    elif(Initials == "LS"):
        path = r"_\HLE_Dataset\LS"
        snake_path = path + r"\Segmentation\3_11500-12000\[1-350]" +r"\3_11500-12000[1-350].snake"
    elif(Initials == "MK"):
        path = r"_\HLE_Dataset\MK"
        snake_path = path + r"\Segmentation\3_14500-15000\[1-475]" +r"\3_14500-15000[1-475].snake"
    elif(Initials == "MS"):
        path = r"_\HLE_Dataset\MS"
        snake_path = path + r"\Segmentation\3_8500-9000\[250-501]" +r"\3_8500-9000[250-501].snake"
    elif(Initials == "RH"):
        path = r"_\HLE_Dataset\RH"
        snake_path = path + r"\Segmentation\2_2500-3000\[1-327]" +r"\2_2500-3000[1-327].snake"
    elif(Initials == "SS"):
        path = r"_\HLE_Dataset\SS"
        snake_path = path + r"\Segmentation\2_2956-3456\[277-501]" +r"\2_2956-3456[277-501].snake"
    elif(Initials == "TM"):
        # Warning: TH has two different Segmentation folders!
        path = r"_\HLE_Dataset\TM"
        snake_path = path + r"\Segmentation\3_6507_7111\[1-605]" +r"\3_6507_7111[1-605].snake" 
    else:
        path = "error"
        
    self_made_image_path = r"_\Segmentation"
    #image_path = path + r"\png\\"
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

        first_img_path = os.path.join(path, "png", FrameNumber + Initials + ".png")
        result_path = os.path.join(path, "Segmentation", "glottis", FrameNumber + Initials + ".png")
        img = cv2.imread(first_img_path, 0)
        snakes = getSnakesAtFrame(snake_path, k)
        binaryImg = np.zeros(img.shape)
        if(snakes == []):
            cv2.imwrite(result_path, binaryImg)
            continue
        max_x = max(snakes, key=itemgetter(0))
        max_x_id = snakes.index(max_x)
        min_x = min(snakes, key=itemgetter(0))
        min_x_id = snakes.index(min_x)

        min_y = min(snakes, key=itemgetter(1))
        min_y_id = snakes.index(min_y)
        max_y = max(snakes, key=itemgetter(1))
        max_y_id = snakes.index(max_y)

        for m in range(min_y_id, max_y_id+1):
            y = snakes[m][1]
            x = snakes[m][0]
            binaryImg[y, x] = 255
        for m in range(max_y_id, len(snakes)):
            y = snakes[m][1]
            x = snakes[m][0]
            binaryImg[y, x] = 255
        binaryImg = drawGlottis(binaryImg, snakes, min_y_id, max_y_id, min_y, max_y)
        cv2.imshow("2", binaryImg.astype(np.uint8) | img)
        cv2.waitKey(0)

        cv2.imwrite(result_path, binaryImg)
