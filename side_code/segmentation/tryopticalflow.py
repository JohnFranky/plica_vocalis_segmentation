from ctypes import windll
from xml.sax.handler import property_lexical_handler
import numpy as np
import cv2 as cv
import os
import data_reorganisation

"""
def checkNeighbours(list, img):
    x = list[0]
    y = list[1]

    img[x][y] = 2
    counter = 1
    if(img[x+1][y] == 1):
        counter+=1
    elif(img[x-1][y] == 1):
        counter+=1
    elif(img[x][y+1] == 1):
        counter+=1
    elif(img[x][y-1] == 1):
        counter+=1
    if counter > 0:
        masks = setNeighbours([i,x,y], masks)
        return True
    else:
        if masks[i][x+1][y] == 1:
            c =  checkNeighbours([i,x+1,y], masks,preds)
            if c:
                return c
        if masks[i][x-1][y] == 1:
            c =  checkNeighbours([i,x-1,y], masks,preds)
            if c:
                return c
        if masks[i][x][y+1] == 1:
            c =  checkNeighbours([i,x,y+1], masks,preds)
            if c:
                return c
        if masks[i][x][y-1] == 1:
            c =  checkNeighbours([i,x,y-1], masks,preds)
            if c:
                return c
        return False
"""
"""
def setNeighbours(list, img):
    x = list[0]
    y = list[1]

    img[x][y] = 0
    if img[x+1][y] == 255:
        img = setNeighbours([x+1,y], img)
    if img[x-1][y] == 255:
        img = setNeighbours([x-1,y], img)
    if img[x][y+1] == 255:
        img = setNeighbours([x,y+1], img)
    if img[x][y-1] == 255:
        img = setNeighbours([x,y-1], img)
    return img


def sortPnts(p0, frame):
    new_pnts = []
    for pnt in p0:

        if(frame[int(pnt[0][1]), int(pnt[0][0])] >= 140):
            new_pnts.append(pnt)
        elif(frame[int(pnt[0][1])-1, int(pnt[0][0])] >= 140):
            new_pnts.append(pnt)
        elif(frame[int(pnt[0][1])+1, int(pnt[0][0])] >= 140):
            new_pnts.append(pnt)
        elif(frame[int(pnt[0][1]), int(pnt[0][0])+1] >= 140):
            new_pnts.append(pnt)
        elif(frame[int(pnt[0][1]), int(pnt[0][0])-1] >= 140):
            new_pnts.append(pnt)
    
    if(len(new_pnts) < 4):
        new_frame = frame.copy()
        for x in range(280, 380):
            for y in range(160, 220):
                if(new_frame[x][y] == 255):
                    flag = True
                    for pnt in new_pnts:
                        if(abs(y - int(pnt[0][1])) + abs(x - int(pnt[0][0])) <= 10):
                            new_frame = setNeighbours([x,y], new_frame)
                            flag = False
                            break

                    if not flag:
                        continue
                    else:
                        new_pnts.append([[np.float32(y), np.float32(x)]])
    return np.array(new_pnts)
    
def find_better_pnts(img):
    woimg = img.copy()
    pnts = []
    for x in range(0, 512):
        for y in range(0, 256):
            if(woimg[x][y] == 255):
                woimg = setNeighbours([x,y], woimg)
                pnts.append(np.array([[x,y]]))
    return np.array(pnts)

if __name__ == "__main__":
    exampleDataset = 0
    path = data_reorganisation.get_path(exampleDataset)[0]
    result = os.path.join(path, "result")
    plica_vocalis = os.path.join(path, "Segmentation", "plica_vocalis")
    path = os.path.join(path, "png")
    images = os.listdir(path)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    counter = 0
    frame1 = cv.imread(os.path.join(path, images[counter]), 0)
    local_window = cv.imread(os.path.join(plica_vocalis, images[counter]), 0)
    old_gray = frame1
    p0 = cv.goodFeaturesToTrack(old_gray, mask = local_window , **feature_params)
    #p1 = find_better_pnts(frame1)




    #cv.imshow('frame1', local_window)
    #cv.waitKey(0)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1)
    while(1):
        counter += 1
        frame = cv.imread(os.path.join(path, images[counter]), 0)
        if len(frame) == 0:
            print('No frames grabbed!')
            break
        frame_gray = frame
        # sort points
        p0 = sortPnts(p0, frame)
        if len(p0) == 0:
            print("NOOOO")
            old_gray = frame_gray.copy()
            continue
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
 
        if p1 is not None:
            good_new = p1
            good_old = p0
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            #mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 2, 0, -1)#color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        #cv.imshow('frame1', frame1)
        k = cv.waitKey(0)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv.destroyAllWindows()
"""


if __name__ == "__main__":
    exampleDataset = 0
    path = data_reorganisation.get_path(exampleDataset)[0]
    result = os.path.join(path, "result")
    path = os.path.join(path, "png")
    images = os.listdir(path)

    # ignore that use own frames
    #cap = cv.VideoCapture(cv.samples.findFile(r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\CF\2_9500_10200.avi"))
    #ret, frame1 = cap.read()

    counter = 0
    frame1 = cv.imread(os.path.join(path, images[counter]), 0)
    #prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    prvs = frame1
    hsv = np.zeros((len(frame1), len(frame1[0]), 3))
    hsv[..., 1] = 255
    while(1):
        counter += 1
        frame2 = cv.imread(os.path.join(path, images[counter]), 0)
        if len(frame2) == 0:
            print('No frames grabbed!')
            break
        #next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        next = frame2
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 11, 3, 7, 1.5, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        img_float32 = np.float32(hsv)
        #lab_image = cv.cvtColor(img_float32, cv.COLOR_RGB2HSV)
        bgr = cv.cvtColor(img_float32, cv.COLOR_HSV2BGR)
        cv.imshow('frame2', bgr)
        cv.imshow('frame', frame2)
        #k = cv.waitKey(30) & 0xff
        cv.waitKey(0)
        #cv.imwrite('opticalfb.png', frame2)
        #cv.imwrite('opticalhsv.png', bgr)
        prvs = next
    cv.destroyAllWindows()
