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

def loadSnakes(path, numFiles):
    snakefile = open(path, mode="r")
    lines = snakefile.readlines()

    currentIndex = 1
    counter = 0
    flip = False
    tmp = []
    snakes = []
    currentIndex = -1
    for line in lines:
        if re.fullmatch("^[0-9][0-9][0-9]\n$", line) or re.fullmatch("^[0-9][0-9]\n$", line) or re.fullmatch("^[0-9]\n$", line):#s[2]):
            currentIndex += 1 
            flip = True

        # MATLAB is one initialized
        if re.fullmatch("^.*\t.*\n$", line):
            if(flip):
                if not (currentIndex == 0):
                    snakes.append(tmp)
                tmp = []
                flip = False
            a, b = line.replace("\t", " ").replace("\n", "").replace(".00", "").split(" ")
            tmp.append([int(a), int(b)])
            counter += 1
    snakes.append(tmp)
    #Get Convex Hull per Image
    #snakeImage = np.zeros((512, 256))
    snakeImages = list() 
    for snake in snakes:
        image = np.zeros((512, 256))
        for x, y in snake:
            image[y, x] = 255
        snakeImages.append(image)
        


    #for i in range(len(snakeImages)):
        #cv2.polylines(snakeImages[i], [np.array(snakes[i][1:]).astype(np.int64)], True, 255, 3)
    #Draw Image containing Convex Hull
    """
    for i in range(len(snakeImages)):
        for j in range(len(snakes[i])):
            elem = snakes[i][j]
            snakeImages[i] = cv2.circle(snakeImages[i], elem, 0, 255, -1)

            #snakeImages[i] = cv2.circle(snakeImages[i], elem, 0, 255, -1)
    """
    return snakeImages

def loadLaserDots(path, numFrames):
    f = scipy.io.loadmat(path)

    points = f['PP'].reshape(numFrames, -1, 2)
    pnts = f['PP']

    dotImage = np.zeros((512, 256), np.uint8)
    dotImages = []
    """
    m =points.shape[0]
    n = points.shape[1]
    coordlist = []
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):

            # This is hacky, we should probably pre-process those matrices to not contain any nans
            # To further accelerate this, we can draw single dots in parallel, and then dilate the image to generate the circles
            dot = points[i][j].astype(np.int64)
            if (dot < 0).any():
                continue
            
            cv2.circle(dotImages[i], dot, radius=2, color=255, thickness=-1)
            coordlist.append(dot)
            cv2.imshow("DotsInfo", dotImages[0])
            #cv2.waitKey(0)
    """
    for i in range(0, anzFrames):
        dotImage = np.zeros((512, 256), np.uint8)
        for x in range(0, len(pnts)):
            for y in range(0, len(pnts[0])):
                dotX = pnts[x][y][0][i].astype(np.int64)
                dotY = pnts[x][y][1][i].astype(np.int64)
                if (dotX < 0 or dotY < 0):
                    continue
                dot = [dotX, dotY]
                cv2.circle(dotImage, dot, radius=0, color=255, thickness=2)
        dotImages.append(dotImage)
        #cv2.imshow("DotsInfo", dotImages[0])
        #cv2.waitKey(0)
    return dotImages

def deleteFailure(x,y,dots):
    #cv2.imshow("Dots", dots)
    #cv2.waitKey(0)
    if(x >= 512 or y >= 256):
        return dots
    dots[x,y] = 0
    if(x+1 <= 255 and dots[x + 1,y] == 255):
        dots = deleteFailure(x+1,y, dots)
    if(x-1 >= 0 and dots[x - 1,y] == 255):
        dots = deleteFailure(x-1,y, dots)
    if(y+1 <= 255 and dots[x,y + 1] == 255):
        dots = deleteFailure(x,y+1, dots)
    if(y-1 >= 0 and dots[x,y - 1] == 255):
        dots = deleteFailure(x,y-1, dots)
    return dots

def laserdotsPostProcessing(laserpoints, counter):
    dotslist = []
    for img in range(0, len(laserpoints)):
        dots = laserpoints[img].copy()
        for y in range(0, len(laserpoints[0]) - (counter + 2)):
            for x in range(0, len(laserpoints[0][0]) - (counter + 2)):
                if(dots[y,x] == 255):
                    rightSide = 0
                    bottomSide = 0
                    diagonalSide = 0
                    for i in range(0, counter):
                        if(dots[y + i,x] == 255):
                            rightSide += 1
                        if(dots[y,x + i] == 255):
                            bottomSide += 1
                        if(dots[y + i,x + i] == 255):
                            diagonalSide += 1
                    
                    if(rightSide >= counter or bottomSide >= counter or diagonalSide >= counter):
                        dots = deleteFailure(y,x,dots)
        
        for y in range(0, len(laserpoints[0])):
            for x in range(245, len(laserpoints[0][0])):
                dots[y,x] = 0
        dotslist.append(dots)
        print("img" + str(img) + " done")

    return dotslist


if __name__ == "__main__":
    Initials = "TM"   
    anzFrames = 605
    if(Initials == "CF"):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\CF"
        snake_path = path + r"\Segmentation\2_9500_10200\[1-701]" +r"\2_9500_10200[1-701].snake"
        laserdots_path = path + r"\results\laserdots\701.mat"
    elif(Initials == "CM"):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\CM"
        snake_path = path + r"\Segmentation\1_200-700\[190-501]" +r"\1_200-700[190-501].snake"
        laserdots_path = path + r"\results\laserdots\312.mat"
    elif(Initials == "DD"):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\DD"
        snake_path = path + r"\Segmentation\Denis2_15518-16018\[1-350]" +r"\Denis2_15518-16018[1-350].snake"
        laserdots_path = path + r"\results\laserdots\350.mat"
    elif(Initials == "FH"):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\FH"
        snake_path = path + r"\Segmentation\4_2000-2650\[1-651]" +r"\4_2000-2650[1-651].snake"
        laserdots_path = path + r"\results\laserdots\651.mat"
    elif(Initials == "LS"):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\LS"
        snake_path = path + r"\Segmentation\3_11500-12000\[1-350]" +r"\3_11500-12000[1-350].snake"
        laserdots_path = path + r"\results\laserdots\350.mat"
    elif(Initials == "MK"):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\MK"
        snake_path = path + r"\Segmentation\3_14500-15000\[1-475]" +r"\3_14500-15000[1-475].snake"
        laserdots_path = path + r"\results\laserdots\475.mat"
    elif(Initials == "MS"):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\MS"
        snake_path = path + r"\Segmentation\3_8500-9000\[250-501]" +r"\3_8500-9000[250-501].snake"
        laserdots_path = path + r"\results\laserdots\252.mat"
    elif(Initials == "RH"):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\RH"
        snake_path = path + r"\Segmentation\2_2500-3000\[1-327]" +r"\2_2500-3000[1-327].snake"
        laserdots_path = path + r"\results\laserdots\327.mat"
    elif(Initials == "SS"):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\SS"
        snake_path = path + r"\Segmentation\2_2956-3456\[277-501]" +r"\2_2956-3456[277-501].snake"
        laserdots_path = path + r"\results\laserdots\225.mat"
    elif(Initials == "TM"):
        # Warning: TH has two different Segmentation folders!
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\TM"
        snake_path = path + r"\Segmentation\3_6507_7111\[1-605]" +r"\3_6507_7111[1-605].snake"
        laserdots_path = path + r"\results\laserdots\605.mat"
    else:
        path = "error"
        
    #image_path = path + r"\png\\"
    
        
    first_img_path = os.path.join(path, "png", "00001" + Initials + ".png")
    laserdotters = cv2.imread(os.path.join(path, "Segmentation", "laserdots", "00001" + Initials + ".png"), 0)
    img = cv2.imread(first_img_path, 0)
    #cv2.imshow("origin", img)
    #cv2.imshow("laserdotters", laserdotters)
    laserdots = loadLaserDots(laserdots_path, anzFrames)
    #cv2.imshow("UnitTest", img | laserdots[0].astype(np.uint8))
    #cv2.waitKey(0)
    #cv2.imshow("origin", img)
    #cv2.imshow("dots", laserdots[0])
    #cv2.waitKey(0)
    #laserdots = laserdotsPostProcessing(laserdots, 6)
    FrameNumberInt = 0
    for dots in laserdots:
        #cv2.imshow("DotsInfo", dots)
        #cv2.waitKey(0)
        dots = dots.astype(np.uint8)
        FrameNumberInt += 1
        if(FrameNumberInt < 10):
            FrameNumber = r"0000"+str(FrameNumberInt)
        elif(FrameNumberInt < 100):
            FrameNumber = r"000"+str(FrameNumberInt)
        elif(FrameNumberInt < 1000):
            FrameNumber = r"00"+str(FrameNumberInt)
        else:
            FrameNumber = r"0"+str(FrameNumberInt)
        print("Saving image " + FrameNumber)
        result_path = os.path.join(path, "Segmentation", "laserdots", FrameNumber + Initials + ".png")
        cv2.imwrite(result_path, dots)
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    """
        laserdots = loadLaserDots(laserdots_path, len(images))
        snakes = loadSnakes(snake_path, len(images)) #array: 475(pngs)x512x256
        for image, snake, laserdot in zip(images, snakes, laserdots): 
        
            cv2.imshow("SnakeInfo",image | snake.astype(np.uint8))
            #cv2.imshow("Laserdots", image | laserdot)
            cv2.waitKey(0)
        images = loadImages(image_path)
        snakes = loadSnakes(snake_path, len(images)) #array: 475(pngs)x512x256
        for image, snake in zip(images, snakes):
            cv2.imshow("SnakeInfo",image | snake.astype(np.uint8))
            cv2.waitKey(0)
    """