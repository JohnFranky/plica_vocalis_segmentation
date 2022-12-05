# coding: utf-8
import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

def get_path(m):
    if(m == 0):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\CF"
        anzFrames = 329
    elif(m == 1):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\CM"
        anzFrames = 311
    elif(m == 2):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\DD"
        anzFrames = 350
    elif(m == 4):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\LS"
        anzFrames = 350
    elif(m == 5):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\MK"
        anzFrames = 474
    elif(m == 6):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\MS"
        anzFrames = 251
    elif(m == 7):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\RH"
        anzFrames = 327
    elif(m == 8):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\SS"
        anzFrames = 225
    elif(m == 9):
        path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\HLE_Dataset\TM"
        anzFrames = 603
    else:
        path = "error"
        anzFrames = -1
    return (path, anzFrames)

def flow2img(flow, BGR=True):
	x, y = flow[..., 0], flow[..., 1]
	hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype = np.uint8)
	ma, an = cv2.cartToPolar(x, y, angleInDegrees=True)
	hsv[..., 0] = (an / 2).astype(np.uint8)
	hsv[..., 1] = (cv2.normalize(ma, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)).astype(np.uint8)
	hsv[..., 2] = 255
	if BGR:
		img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	else:
		img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	return img


#initials = "CF"

m = -1
print("optical-flow")
while (m < 9):
	startT = time.perf_counter()
	m+=1
	if m == 3:
		continue
	path, max = get_path(m)
	all_masks = path + r"\Segmentation\all_masks"
	save_pth = path + r"\Segmentation\opticalflow\SF\Raw"
	#flows = list()
	length = max -2
	os.mkdir(path + r"\Segmentation\opticalflow\SF")
	os.mkdir(path + r"\Segmentation\opticalflow\SF\Raw")
	path = path +r"\png"
	images = os.listdir(path)
	for k in range (0, length):
		if k%10 == 0:
			print("Working on frame " +str(k))
		im1 = cv2.imread(path + "\\" + images[k])
		im2 = cv2.imread(path + "\\" + images[k+1])
		gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

		"""
		dtvl1 = cv2.optflow.DualTVL1OpticalFlow_create(tau = 0.25,theta = 0.15,nscales = 5,warps = 10,epsilon = 0.005,innnerIterations = 30,outerIterations = 10,scaleStep = 0.8,gamma = 0.0,medianFiltering = 5)
		#different paras!!!!!!!!!!
		flowDTVL1 = dtvl1.calc(gray1, gray2, None)
		"""
		
		flowSF = cv2.optflow.calcOpticalFlowSF(im1, im2, 3, 5, 5)
		cv2.imwrite(os.path.join(save_pth, images[k+1]), flow2img(flowSF, False))
		#flows.append(flow2img(flowDTVL1, False))

	endT = time.perf_counter()
	print(endT - startT, "s")

"""
print("post-processing")

new_flows = np.full(np.array(flows).shape,255, dtype="uint8")
for i in range(len(flows)):
	if i%10 == 0:
		print("Working on frame " +str(i))
	mask = cv2.imread(all_masks + "\\" + images[i])
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	for x in range(len(flows[i])):
		for y in range(len(flows[i][0])):
			if(mask[x][y] == 2 or mask[x][y] == 3):
				marked = 0
				for j in range(-5,6):
					if mask[x][y+j] == 1:
						marked += 1
						break
				if not (marked == 0):
					continue
				intensity = int(flows[i][x][y][0]) + int(flows[i][x][y][1]) + int(flows[i][x][y][2])	#this is not really a metric
				if(intensity <= 255*2 + 125):#75
					new_flows[i][x][y] = flows[i][x][y]

for i in range(len(flows)):
	cv2.imshow("Pre",cv2.imread(path + "\\" + images[i]))
	cv2.imshow("Post",cv2.imread(path + "\\" + images[i+1]))
	cv2.imshow("OpticalFLow",flows[i])
	cv2.imshow("OpticalFLowFiltered",new_flows[i])
	cv2.waitKey(0)
"""
