import numpy as np
import cv2
import os


def sharpen_image(path):
    image = cv2.imread(path)
    #cv2.imshow("Original", image)
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    #cv2.imshow('Sharp', image_sharp)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    save_path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\Segmentation\sharpened.png"
    result=cv2.imwrite(save_path, image_sharp)
    return save_path




def Filter_Shiny(path):
   # load the image, and blur it
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    blurredMedian = image
    for i in range(0, 5):
        blurredMedian = cv2.medianBlur(blurredMedian, 3)
    #cv2.imshow("Median", blurredMedian)
    # threshold the image to reveal light regions in the
    # blurred image
    #thresh = cv2.threshold(blurredMedian, 200, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("Original", image)
    #cv2.imshow("blurr", thresh)
    #cv2.waitKey(0)
    save_path = r"E:\Eigene Dateien Jonathan\studium\Letztes_Semester\Bachlor\Segmentation\blurred.png"
    result=cv2.imwrite(save_path, blurredMedian)
    return save_path

def edgeDetection(path, pic):
    if pic == True:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = np.uint8(path)
    blurred = cv2.GaussianBlur(image, (3,3), 0)

    edges = cv2.Canny(image=blurred, threshold1=100, threshold2=200) # Canny Edge Detection
    #cv2.imshow('Original', image)
    cv2.imshow('Canny Edge Detection', edges)
    #cv2.waitKey(0)
    return edges

def contourNoise(path, pic):
    # Load image, grayscale, Otsu's threshold
    if pic == True:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = np.uint8(path)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours and filter using contour area
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            cv2.drawContours(thresh, [c], -1, 0, -1)
            cv2.drawContours(image, [c], -1, (36,255,12), -1)

    result = 255 - thresh
    #cv2.imshow("image", image) 
    #cv2.imshow("thresh", thresh) 
    #cv2.imshow("result", result) 
    return result


"""
def alternateSeedPoints(snake_path, first_img_path, FrameNumberInt, offsetShiny, first_image):
    offset = -60
    lengthlist = []
    while(offset < 60):
        tuple = getSeedPoints(snake_path, first_img_path, FrameNumberInt, offsetShiny + offset)
        binaryImg = regionGrow(first_image, [tuple[0]], tuple[1])
        length = sum(sum(binaryImg))
        lengthlist.append((length, binaryImg, tuple))
        offset += 3
    i = 0
    lenge = len(lengthlist)
    while(i < lenge):
        if(lengthlist[i][0] < 100 or lengthlist[i][0] > 5000):
            del(lengthlist[i])
            lenge -= 1
            continue
        
        if(contourCheck(lengthlist[i][1])):
            del(lengthlist[i])
            lenge -= 1
            continue
        i +=1
    
    return lengthlist
"""
path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\MK"
#test2 = cv2.imread(os.path.join(path, "png","00001MS.png"),0)
img = edgeDetection(os.path.join(path, "png","00001MK.png"), True)
cv2.imwrite(os.path.join(path, "Segmentation", "image.png"), img)
