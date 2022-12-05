from ctypes.wintypes import POINT
import cv2
from matplotlib import offsetbox
from matplotlib.image import imread
import numpy as np
import sys
import re
import os
from operator import itemgetter
from PIL import Image, ImageFilter
from torch import true_divide
import math
import random

doppelt = True #sets 2 seed points instead of one
nosnake = False
#forceLowerhalfPnt = False

def loadImages(path):
    images = list()
    number_files = len(os.listdir(path))

    for i in range(1, number_files):
        #x = path + '{0:05d}'.format(i) + ".png"
        images.append(cv2.imread(path + '{0:05d}'.format(i) + ".png", 0))

    return images

def find_nearest(array, value):
    aux = []
    for valor in array:
        aux.append(abs(int(value)-int(valor)))
    return aux.index(min(aux))

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

def alternateLines(xLeft, xRigth, yTop, yBot):
    lineLeft = []
    for y in range(yTop, yBot +1):
        lineLeft.append(Point(y ,xLeft))
    
    lineRight = []
    for y in range(yTop, yBot +1):
        lineRight.append(Point(y ,xRigth))
    
    return (lineLeft, lineRight)

def rebuildLines(GrayLeft, GrayRigth, y, bot = False):
    k = 0
    while k < len(GrayLeft):
        i = GrayLeft[k]
        if bot and i.x < y:
            GrayLeft.remove(i)
            continue
        elif not bot and i.x > y:
            GrayLeft.remove(i)
            continue
        k += 1
    k = 0
    while k < len(GrayRigth):
        i = GrayRigth[k]
        if bot and i.x < y:
            GrayRigth.remove(i)
            continue
        elif not bot and i.x > y:
            GrayRigth.remove(i)
            continue
        k += 1
    
    return (GrayLeft, GrayRigth)


def getSeedPoints(pathSnakes, pathPicture, frame_number, offsetShiny, forceLowerhalfPnt = False, forceMiddlePnt = False):
    snakes = getSnakesAtFrame(pathSnakes, frame_number)
    if(snakes == []):
        global nosnake
        nosnake = True
    
    image = cv2.imread(pathPicture)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("ori", image)
    if(not nosnake):
        max_x = max(snakes, key=itemgetter(0))
        max_x_id = snakes.index(max_x)
        min_x = min(snakes, key=itemgetter(0))
        min_x_id = snakes.index(min_x)

        min_y = min(snakes, key=itemgetter(1))
        min_y_id = snakes.index(min_y)
        max_y = max(snakes, key=itemgetter(1))
        max_y_id = snakes.index(max_y)

        linepnts = []
        if(doppelt):
            linepntsLeft = []
            linepntsRight = []
            linepntsGrayLeft = []
            linepntsGrayRight = []
        offset = 16
        if(len(snakes) < 15):
            offset = 0
        pxlOffset = 6
        diff_topToSide = min_x_id - min_y_id

        for i in range(min_y_id + offset, max_y_id): #- int(offset/4)):
            linepnts.append(Point(snakes[i][1], snakes[i][0]- pxlOffset))
            if(doppelt):
                linepntsLeft.append(Point(snakes[i][1], snakes[i][0]- pxlOffset))
        
        for i in range(max_y_id , len(snakes) - offset):
            linepnts.append(Point(snakes[i][1], snakes[i][0] + pxlOffset))
            if(doppelt):
                linepntsRight.append(Point(snakes[i][1], snakes[i][0]+ pxlOffset))
        
        
    else:
        lines = alternateLines(193, 225, 190, 260)
        linepntsLeft = lines[0]
        linepntsRight = lines[1]
        linepntsGrayLeft = []
        linepntsGrayRight = []
    #if(forceLowerhalfPnt):
    res = rebuildLines(linepntsLeft, linepntsRight, 335, bot = forceLowerhalfPnt)
    linepntsLeft = res[0]
    linepntsRight = res[1]
    pnts = len(linepntsLeft) + len(linepntsRight)
    if(pnts < 30):
        lines = alternateLines(193, 225, 190, 260)
        linepntsLeft = lines[0]
        linepntsRight = lines[1]
        linepntsGrayLeft = []
        linepntsGrayRight = []
    if(forceLowerhalfPnt):
        lines = alternateLines(175, 195, 360, 375)
        linepntsLeft = lines[0]
        linepntsRight = lines[1]
        linepntsGrayLeft = []
        linepntsGrayRight = []
    if(forceMiddlePnt):
        lines = alternateLines(175, 195, 330, 355)
        linepntsLeft = lines[0]
        linepntsRight = lines[1]
        linepntsGrayLeft = []
        linepntsGrayRight = []

    #Get grey values of such points and calculate threshold
    linepntsGrayValue = []
    avg = 0
    avgLeft = 0
    avgRight = 0
    if(not doppelt):
        pnts = len(linepnts)
        for i in linepnts:
            gray = image[i.x, i.y]
            if(gray <= 230 and gray >= 5):
                avg += gray
                linepntsGrayValue.append(gray)
            else:
                if(len(linepntsGrayValue) == 0):
                    del linepnts[0]
                else:
                    linepntsGrayValue.append(linepntsGrayValue[len(linepntsGrayValue)-1])
    if(doppelt):
        pnts = len(linepntsLeft) + len(linepntsRight)
        for i in linepntsLeft:
            gray = image[i.x, i.y]
            if(gray <= 230 and gray >= 5):
                linepntsGrayLeft.append(gray)
                avgLeft += gray
            else:
                if(len(linepntsGrayLeft) == 0):
                    del linepntsLeft[0]
                else:
                    linepntsGrayLeft.append(linepntsGrayLeft[len(linepntsGrayLeft)-1])
        
        for i in linepntsRight:
            gray = image[i.x, i.y]
            if(gray <= 230 and gray >= 20):
                linepntsGrayRight.append(gray)
                avgRight += gray
            else:
                if(len(linepntsGrayRight) == 0):
                    del linepntsRight[0]
                else:
                    linepntsGrayRight.append(linepntsGrayRight[len(linepntsGrayRight)-1])
        if(len(linepntsGrayLeft) < 6 or len(linepntsGrayRight) < 6):
            offset = 0
            if(len(linepntsGrayLeft) < 6 ):
                for i in range(min_y_id + offset, len(snakes) -offset):
                    if(doppelt):
                        linepntsLeft.append(Point(snakes[i][1], snakes[i][0]- pxlOffset))
                for ix in range(len(linepntsGrayLeft), len(linepntsLeft)):
                    i = linepntsLeft[ix]
                    gray = image[i.x, i.y]
                    if(gray <= 230 and gray >= 20):
                        avgLeft += gray
                        linepntsGrayLeft.append(gray)
                    else:
                        if(len(linepntsGrayLeft) == 0):
                            del linepntsLeft[0]
                        else:
                            linepntsGrayLeft.append(linepntsGrayLeft[len(linepntsGrayLeft)-1])
            else:
                for i in range(max_x_id + offset, len(snakes) -offset):
                    if(doppelt):
                        linepntsRight.append(Point(snakes[i][1], snakes[i][0]+ pxlOffset))
                for ix in range(len(linepntsGrayRight), len(linepntsRight)):
                    i = linepntsRight[ix]
                    gray = image[i.x, i.y]
                    if(gray <= 230 and gray >= 20):
                        linepntsGrayRight.append(gray)
                        avgRight += gray
                    else:
                        if(len(linepntsGrayRight) == 0):
                            del linepntsRight[0]
                        else:
                            linepntsGrayRight.append(linepntsGrayRight[len(linepntsGrayRight)-1])
    avg /= pnts
    avgLeft /= len(linepntsGrayLeft)
    avgRight /= len(linepntsGrayRight)
    highest = 0
    lowest = 0
    anzHL = 8 #int(len(linepntsGrayValue)/10)
    if(not doppelt):
        avgOffset = int(avg) + offsetShiny
        idx = find_nearest(linepntsGrayValue, avgOffset)
        seed_point = linepnts[idx]
    else:
        avgOffset = int((avgLeft + avgRight)/2) + offsetShiny
        idx1 = find_nearest(linepntsGrayLeft, avgOffset)
        idx2 = find_nearest(linepntsGrayRight, avgOffset)
        if(avgOffset - linepntsGrayLeft[idx1] > 10):
            
            avgOffset = int(linepntsGrayLeft[idx1]) + int(random.uniform(1, 10)) #int(avgLeft) + (linepntsGrayLeft[idx1] - int(avgLeft))/2
            idx1 = find_nearest(linepntsGrayLeft, avgOffset)
            idx2 = find_nearest(linepntsGrayRight, avgOffset)
        if(avgOffset - linepntsGrayRight[idx2] > 10):
            avgOffset = int(linepntsGrayRight[idx2]) + int(random.uniform(1, 10)) #int(avgRight) + (linepntsGrayRight[idx2] - int(avgRight))/2
            idx1 = find_nearest(linepntsGrayLeft, avgOffset)
            idx2 = find_nearest(linepntsGrayRight, avgOffset)
        seed_point = [linepntsLeft[idx1], linepntsRight[idx2]]
    if(doppelt):
        if(len(linepntsGrayLeft) > len(linepntsGrayRight)):
            linepntsGrayValue = linepntsGrayLeft
        else:
            linepntsGrayValue = linepntsGrayRight
    for i in range (0, anzHL):
        if(len(linepntsGrayValue) <= 1):
            anzHL -= i-1
            break
        x = max(linepntsGrayValue)
        highest += x
        linepntsGrayValue.remove(x)

        x = min(linepntsGrayValue)
        lowest += x
        linepntsGrayValue.remove(x)
    highest /= anzHL
    lowest /= anzHL
    threshold = int((highest - lowest)/5)
    """
    Visualization:
    #cv2.imshow("original", image)
    image = np.zeros((512, 256))
    for x in linepnts:
        image[x.x, x.y] = 255
    #cv2.imshow("linepnts", image)
    #cv2.waitKey(0)
    """
    return (seed_point,threshold)
    

def correctImage(topSide,binImg, middleSide):
    
    leftCorner = 1000
    rightCorner = -1
    topCorner = 1000
    for x in range(0, 256):
        for y in range(0, 512):
            if((binImg[y,x] == 255 or binImg[y,x] == 1.0)):
                if y < topCorner:
                    topCorner = y
                if x < leftCorner:
                    leftCorner = x
                if x > rightCorner:
                    rightCorner = x
    offset = 5
    for x in range(0, 256):
        for y in range(0, 512):
            if((middleSide[y,x] == 255 or middleSide[y,x] == 1.0)):
                if x < leftCorner - offset:
                    middleSide[y,x] = 0.0
                if x > rightCorner + offset + 5:
                    middleSide[y,x] = 0.0
                """
                if y < topCorner - offset - 5:
                    middleSide[y,x] = 0
                """
    full = binImg + topSide + middleSide
    #cv2.imshow('1',full)
    lineLeft = []
    lineRigth = []
    for y in range(300, 371):
        for x in range(120, 180):
            if((full[y,x] == 255 or full[y,x] == 1.0)):
                lineLeft.append([x,y])
                break
        
        for x in range(230, 180, -1):
            if((full[y,x] == 255 or full[y,x] == 1.0)):
                lineRigth.append([x,y])
                break
    
    max_x_R = max(lineRigth, key=itemgetter(0))
    max_x_id_R = lineRigth.index(max_x_R)
    min_x_L = min(lineLeft, key=itemgetter(0))
    min_x_id_L = lineLeft.index(min_x_L)
    lineLeft.remove(lineLeft[min_x_id_L])
    lineRigth.remove(lineRigth[max_x_id_R])
    for i in range(0, len(lineLeft)):
        min_x = min(lineLeft, key=itemgetter(0))
        min_x_id = lineLeft.index(min_x)
        if(abs(lineLeft[min_x_id][1] - min_x_L[1]) > 50):
            full = cv2.line(full, lineLeft[min_x_id], min_x_L, 255, 1)
            break
        else:
            lineLeft.remove(lineLeft[min_x_id])
    #cv2.imshow('12',full)
    for i in range(0, len(lineRigth)):
        max_x = max(lineRigth, key=itemgetter(0))
        max_x_id = lineRigth.index(max_x)
        if(abs(lineRigth[max_x_id][1] - max_x_R[1]) > 50):
            full = cv2.line(full, lineRigth[max_x_id], max_x_R, 255, 1)
            break
        else:
            lineRigth.remove(lineRigth[max_x_id])
    #cv2.imshow('123',full)
    
    return full




def smooth_raster_lines(im, filterRadius, filterSize, sigma):
    smoothed = np.zeros_like(im)
    contours, hierarchy = cv2.findContours(im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    for countur_idx, contour in enumerate(contours):
        len_ = len(contour) + 2 * filterRadius
        idx = len(contour) - filterRadius
        x = []
        y = []    
        for i in range(len_):
            x.append(contour[(idx + i) % len(contour)][0][0])
            y.append(contour[(idx + i) % len(contour)][0][1])

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        xFilt = cv2.GaussianBlur(x, (filterSize, filterSize), sigma, sigma)
        xFilt = [q[0] for q in xFilt]
        yFilt = cv2.GaussianBlur(y, (filterSize, filterSize), sigma, sigma)
        yFilt = [q[0] for q in yFilt]


        smoothContours = []
        smooth = []
        for i in range(filterRadius, len(contour) + filterRadius):
            smooth.append([xFilt[i], yFilt[i]])

        smoothContours = np.asarray([smooth], dtype=np.int32)


        color = (0,0,0) if hierarchy[countur_idx][3] > 0 else (255,255,255)
        cv2.drawContours(smoothed, smoothContours, 0, color, -1)
    
    return(smoothed)

def filling(binaryImg):
    idxBeginn = -1
    idxEnd = -1
    black = True
    for y in range(0, 512):
        for x in range(0, 256):
            if(binaryImg[y, x] == 0):
                black = True
                continue
            else:
                if(black == False):
                    idxBeginn = x
                else:
                    idxEnd = x
                black = False
            if(not(idxBeginn == -1) and not(idxEnd == -1)):
                for i in range(idxBeginn, idxEnd):
                    binaryImg[y,i] = 255
                idxBeginn = idxEnd
                idxEnd = -1
        idxBeginn = -1
        idxEnd = -1
    return binaryImg

def cutGlottis(binaryImg, snakes, min_y_id, max_y_id, min_y, max_y):
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
                    binaryImg[snake[i][1], snake[i][0]] = 0
                    break
                else:
                    x0 = snake[i][0]
                    x1 = snake[i+1][0]
                    binaryImg[snake[i][1], snake[i][0]] = 0
                    for diff in range(x0, x1):
                        binaryImg[snake[i][1], diff] = 0
        #cv2.imshow("Line", binaryImg)
        #cv2.waitKey(0)  
    return binaryImg

def drawSnakeLines(pathSnakes, binaryImg, frame_number):
    snakes = getSnakesAtFrame(pathSnakes, frame_number)
    if(snakes == []):
        return binaryImg
    #cv2.imshow("Origin", binaryImg)
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
    #cv2.imshow("GlottisContours", binaryImg)
    binaryImg = filling(binaryImg)
    #cv2.imshow("Filling", binaryImg)
    #binaryImg = cutGlottis(binaryImg, snakes, min_y_id, max_y_id, min_y, max_y)
    #cv2.imshow("cutGlottis", binaryImg)
    return (binaryImg, snakes, min_y_id, max_y_id, min_y, max_y)

def contourCheck(binaryImg):
    pnts = []
    for y in range(0,512):
        for x in range(0,256):
            if(binaryImg[y,x] == 255 or binaryImg[y,x] == 1.0):
                pnts.append([y,x])
    if(len(pnts) == 0):
        return True
    
    
    max_x = max(pnts, key=itemgetter(1))[1]
    min_x = min(pnts, key=itemgetter(1))[1]

    min_y = min(pnts, key=itemgetter(0))[0]
    max_y = max(pnts, key=itemgetter(0))[0]
    
    width = []
    for y in range (min_y, max_y+1):
        avg = 0
        for pnt in pnts:
            if(pnt[0] == y):
                avg += 1
        width.append(avg)
    widthInt = sum(width)/len(width)

    length = []
    for y in range (min_x, max_x+1):
        avg = 0
        for pnt in pnts:
            if(pnt[1] == y):
                avg += 1
        length.append(avg)
    lengthInt = sum(length)/len(length)


    if(lengthInt > 2 * widthInt):
        return True
    if(widthInt > 2* lengthInt):
        return True
    return False

def topSideCheck(snake_path, binaryImg, frame_number):
    #check if a pixel colored is placed higher thatn the top most snake pixel -> error
    snakes = getSnakesAtFrame(snake_path, frame_number)
    snakesSorted= sorted(snakes , key=lambda k: [k[1], k[0]])
    offset = 20 #MS = -2 #MK = 5                                                                                  #Offset in topSide: How much tolerance should there be between top point glottis and top point vocalis? {MK: 5, MS: 1}
    """
    min_y_snakes = min(snakes, key=itemgetter(1))[1]
    min_y_binImg = -1
    for y in range(0,512):
        for x in range(0,256):
            if(binaryImg[y,x] == 255 or binaryImg[y,x] == 1.0):
                min_y_binImg = y
                break
        if not (min_y_binImg == -1):
            break
    if(min_y_binImg < min_y_snakes-offset):
        return True
    """
    offset = 8                                                                                               #Offset from right picture rand
    for x in range(255-offset, 256):
        for y in range(0, 512):
            if(binaryImg[y,x] == 255 or binaryImg[y,x] == 1.0):
                return True
    
    for y in range(166, 275):
        if(binaryImg[y,170] == 255 or binaryImg[y,170] == 1.0):
                return True
    
    for x in range(172, 240):
        if(binaryImg[156,x] == 255 or binaryImg[156,x] == 1.0):
                return True
        if(binaryImg[288, x] == 255 or binaryImg[288, x] == 1.0):
                return True
    
    
    """
    if((binaryImg[255,128] == 255 or binaryImg[255,128] == 1.0 and binaryImg[255,127] == 255 or binaryImg[255,127] == 1.0) or (binaryImg[255,173] == 255 or binaryImg[255,173] == 1.0 and binaryImg[255,174] == 255 or binaryImg[255,174] == 1.0)  or (binaryImg[255,181] == 255 or binaryImg[255,181] == 1.0 and binaryImg[255,182] == 255 or binaryImg[255,182] == 1.0)):
        return True
    criticalAreaCounter = 0
    for x in range(120, 140):
        if(binaryImg[255,x] == 1.0 or binaryImg[255,x] == 255):
            criticalAreaCounter += 1
    for x in range(175, 195):
        if(binaryImg[255,x] == 1.0 or binaryImg[255,x] == 255):
            criticalAreaCounter += 1
    if(criticalAreaCounter > 6):
        return True
    """

    """
    offset = 5
    max_x = max(snakes, key=itemgetter(0))
    min_x = min(snakes, key=itemgetter(0))

    min_y = min(snakes, key=itemgetter(1))
    max_y = max(snakes, key=itemgetter(1))
    pxlerrors = 0
    for snake in snakes:
        if(snake[1] <= min_y[1]+offset):
            continue
        elif(snake[1]+offset >= max_y[1]):
            continue
        else:
            if snake[0] < min_y[0]:
                if(binaryImg[snake[1],snake[]] == 255 or binaryImg[y,255] == 1.0):        
            else:
                
    """           
    return False

def leftCheck(binaryImg):
    """
    offset = 144
    for y in range(0,512):
        for x in range(0,offset):
            if(binaryImg[y,x] == 255 or binaryImg[y,x] == 1.0):
                return True
    """
    return False

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x
    def getY(self):
        return self.y

def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))

def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
            Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1), Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects

def regionGrow(img,seeds, thresh,p = 1, activateLine = False, activatebottom = False):
    doubleSeed = False
    if(len(seeds) > 1):
        doubleSeed = True
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    
    for seed in seeds:
        if(type(seed) == type(Point(1,2))):
            seedList.append(seed)
        else:
            seedList.append(Point(seed[0], seed[1]))
    origin = seedList[0]
    if(doubleSeed):
        origin1 = seedList[0]
        origin2 = seedList[1]
    
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            """
            if(activateLine and ((tmpY >= 208 or tmpY <= 150) or tmpX < 274)):
                continue

            if(activatebottom and (tmpY >= 212 or tmpY <= 144)):
                continue

            if(tmpX <= 160 and tmpY > 360):
                continue
            """
            if(doubleSeed):
                length1 = math.sqrt( math.pow((tmpX - origin1.x),2) + math.pow((tmpY - origin1.y),2) )
                length2 = math.sqrt( math.pow((tmpX - origin2.x),2) + math.pow((tmpY - origin2.y),2) )
              
                if(length1 < length2):
                    origin = origin1
                else:
                    origin = origin2
            
            grayDiff = getGrayDiff(img,origin,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark

def fastEndCheck(first_image, tuple, length):
    if(length < 500):
        finMax = 10
    elif(length < 1000):
        finMax = 8
    else:
        finMax = 6 
    for i in range(1, finMax):
            tmpImg = regionGrow(first_image, tuple[0], tuple[1]+i)
            tmpLength = sum(sum(tmpImg))
            #cv2.imshow("ye", tmpImg)
            #cv2.waitKey(0)
            if(tmpLength - length > 5500):
                return True
            else:
                length = tmpLength
    return False



if __name__ == "__main__":
    Initials = "TM"
    FrameNumberStart = 51
    FrameNumberEnd = 605 #number of last frame included
    #FrameNumberInt = int(FrameNumber) 
    offsetShiny = -6 #offset to favor brighter pnts in seed evaluation. Maybe this needs to be changed in regards to different data #MK=5
    if(Initials == "CF"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\CF"
        snake_path = path + r"\Segmentation\2_9500_10200\[1-701]" +r"\2_9500_10200[1-701].snake"
    elif(Initials == "CM"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\CM"
        snake_path = path + r"\Segmentation\1_200-700\[190-501]" +r"\1_200-700[190-501].snake"
    elif(Initials == "DD"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\DD"
        snake_path = path + r"\Segmentation\Denis2_15518-16018\[1-350]" +r"\Denis2_15518-16018[1-350].snake"
    elif(Initials == "FH"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\FH"
        snake_path = path + r"\Segmentation\4_2000-2650\[1-651]" +r"\4_2000-2650[1-651].snake"
    elif(Initials == "LS"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\LS"
        snake_path = path + r"\Segmentation\3_11500-12000\[1-350]" +r"\3_11500-12000[1-350].snake"
    elif(Initials == "MK"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\MK"
        snake_path = path + r"\Segmentation\3_14500-15000\[1-475]" +r"\3_14500-15000[1-475].snake"
    elif(Initials == "MS"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\MS"
        snake_path = path + r"\Segmentation\3_8500-9000\[250-501]" +r"\3_8500-9000[250-501].snake"
    elif(Initials == "RH"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\RH"
        snake_path = path + r"\Segmentation\2_2500-3000\[1-327]" +r"\2_2500-3000[1-327].snake"
    elif(Initials == "SS"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\SS"
        snake_path = path + r"\Segmentation\2_2956-3456\[277-501]" +r"\2_2956-3456[277-501].snake"
    elif(Initials == "TM"):
        path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\HLE_Dataset\TM"
        snake_path = path + r"\Segmentation\3_6507_7111\[1-605]" +r"\3_6507_7111[1-605].snake" 
    else:
        path = "error"
        
    self_made_image_path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\Segmentation"
    result_path = r"\Segmentation" + r"\plica_vocalis"
    image_path = path + r"\png\\"
    for k in range (FrameNumberStart, FrameNumberEnd + 1):
        nosnake = False
        FrameNumberInt = k
        if(FrameNumberInt < 10):
            FrameNumber = r"0000"+str(FrameNumberInt)
        elif(FrameNumberInt < 100):
            FrameNumber = r"000"+str(FrameNumberInt)
        elif(FrameNumberInt < 1000):
            FrameNumber = r"00"+str(FrameNumberInt)
        else:
            FrameNumber = r"0"+str(FrameNumberInt)
        print("#####################################")
        print("Looking at frame "+FrameNumber)
        first_img_path = image_path + FrameNumber + r".png"
        
        #Loading images
        first_image = cv2.imread(first_img_path, 0)
        cv2.imshow('Original',first_image)
    


        tuple = getSeedPoints(snake_path, first_img_path, FrameNumberInt, offsetShiny, False) #(seed_value, theshold)

        #threshold= cv2.threshold(sharpend_image, 120, 220, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    



        if(doppelt):
            binaryImg1 = regionGrow(first_image, [tuple[0][0]], tuple[1])
            binaryImg2 = regionGrow(first_image, [tuple[0][1]], tuple[1])
            length1 = sum(sum(binaryImg1))
            length2 = sum(sum(binaryImg2))
            #cv2.imshow('Offset: Start 1',binaryImg1)
            #cv2.imshow('Offset: Start 2',binaryImg2)
            if(length1 > 3500 or length1 < 10 or (length1 < 500 and contourCheck(binaryImg1)) or fastEndCheck(first_image, tuple, length1) or topSideCheck(snake_path, binaryImg1, FrameNumberInt) or length2 > 3500 or length2 < 10 or (length2 < 500 and contourCheck(binaryImg2)) or topSideCheck(snake_path, binaryImg2, FrameNumberInt) or fastEndCheck(first_image, tuple, length2)):
                i = 5
                length1 = 10000
                length2 = 10000
                old1 = 10000
                old2 = 10000
                countEqual = 0
                while(length1 > 1850 or length2 > 1850 or length1 > 100*length2 or length2 > 100*length1):
                    tuple1 = getSeedPoints(snake_path, first_img_path, FrameNumberInt, offsetShiny + i, False)
                    binaryImg1 = regionGrow(first_image, [tuple1[0][0]], tuple1[1])
                    length1 = sum(sum(binaryImg1))
                    binaryImg2 = regionGrow(first_image, [tuple1[0][1]], tuple1[1])
                    length2 = sum(sum(binaryImg2))
                    #cv2.imshow('Offset: Higher 1',binaryImg1)
                    #cv2.imshow('Offset: Higher 2',binaryImg2)
                    #cv2.waitKey(0)
                    if((length1 <= 4.0 or length2 <= 4.0) and i > 45):
                        tuple1 = getSeedPoints(snake_path, first_img_path, FrameNumberInt, offsetShiny + i-4)
                        binaryImg1 = regionGrow(first_image, tuple1[0], tuple1[1])
                        length1 = sum(sum(binaryImg1))
                        break
                    if(old1 == length1 and old2 == length2):
                        countEqual += 1
                    
                    if(countEqual > 6):
                        break
                    i += 4
                    old1 = length1
                    old2 = length2
                    if(i > 85):
                        length1 = -1
                        break
                    if(length1 > 1850 or length2 > 1850):
                        continue

                    if((length1 < 500 and contourCheck(binaryImg1)) or fastEndCheck(first_image, tuple1, length1) or topSideCheck(snake_path, binaryImg1, FrameNumberInt) or (length2 < 500 and contourCheck(binaryImg2)) or fastEndCheck(first_image, tuple1, length2) or topSideCheck(snake_path, binaryImg2, FrameNumberInt) or (length1 < 2 or length2 < 2) or length1 > (10 * length2) or length2 > (10 * length1)):  
                        length1 = 10000
                        length2 = 10000

                tuple = tuple1
            binaryImg = regionGrow(first_image, tuple[0], tuple[1])
            length = sum(sum(binaryImg))
            
        
        #cv2.imshow('Before_RG',binaryImg)
        print("Scanned source and seeds")
       
        for i in range(1, 100):
            tmpImg = regionGrow(first_image, tuple[0], tuple[1]+i)
            tmpLength = sum(sum(tmpImg))
            #cv2.imshow('RG',tmpImg)
            #cv2.waitKey(0)
            if(tmpLength - length > 5000 or topSideCheck(snake_path, tmpImg, FrameNumberInt)):
                break
            elif leftCheck(tmpImg):
                break
            else:
                length = tmpLength
                binaryImg = tmpImg
        
        #binaryImg = correctImage(topSide, binaryImg, bottomSide)
        binaryImg
        
        #cv2.imshow('Result of RG',binaryImg)
        print("Region Growing Completed")
        if(nosnake):
            binaryImg = filling(binaryImg)
            binaryImg = np.uint8(binaryImg)
            binaryImg = smooth_raster_lines(binaryImg, 10, 21, 20)
            #cv2.imshow('Smoothedoutlines',binaryImg)
        else:
            bigTuple = drawSnakeLines(snake_path, binaryImg, FrameNumberInt)
            binaryImg = bigTuple[0]
            #cv2.imshow('FullContour',binaryImg)
            binaryImg = np.uint8(binaryImg)
            binaryImg = smooth_raster_lines(binaryImg, 10, 21, 20)
            #cv2.imshow('Smoothedoutlines',binaryImg)
            binaryImg = cutGlottis(binaryImg, bigTuple[1], bigTuple[2], bigTuple[3], bigTuple[4],bigTuple[5])
        print("Polishing completed")
        #cv2.imshow("cutGlottis", binaryImg)
        #cv2.waitKey(0)

        cv2.imwrite(path + result_path + r"\\" + FrameNumber + r".png", binaryImg)
