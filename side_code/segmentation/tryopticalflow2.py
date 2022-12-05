from ctypes import windll
from xml.sax.handler import property_lexical_handler
import numpy as np
import cv2 as cv
import os
import data_reorganisation

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

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1)
    while(1):
        counter += 1
        frame = cv.imread(os.path.join(path, images[counter]), 0)
        if len(frame) == 0:
            print('No frames grabbed!')
            break
        frame_gray = frame
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 4, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        #cv.imshow('frame1', frame1)
        k = cv.waitKey(0)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv.destroyAllWindows()