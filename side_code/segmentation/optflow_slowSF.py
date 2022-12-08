# coding: utf-8
import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

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

initials = "SS"

path = r"_\HLE_Dataset" + "\\" + initials +r"\png"
plica_vocalis = r"_\HLE_Dataset" + "\\" + initials + r"\Segmentation\plica_vocalis"
glottis = r"_\HLE_Dataset" + "\\" + initials + r"\Segmentation\glottis"
all_masks = r"_\HLE_Dataset" + "\\" + initials + r"\Segmentation\all_masks"
images = os.listdir(path)
flows = list()
length = int((len(images)-2)/4)
print("optical-flow")
startT = time.perf_counter()
for k in range (0, length):
	if k%10 == 0:
		print("Working on frame " +str(k))
	im1 = cv2.imread(path + "\\" + images[k])
	im2 = cv2.imread(path + "\\" + images[k+1])
	gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	flowSF = cv2.optflow.calcOpticalFlowSF(im1, im2, 3, 5, 5)
	flows.append(flow2img(flowSF, False))


endT = time.perf_counter()
print(endT - startT, "s")

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
				"""
				marked = 0
				for j in range(-5,6):
					if mask[x][y+j] == 1:
						marked += 1
						break
				if not (marked == 0):
					continue
				"""
				intensity = int(flows[i][x][y][0]) + int(flows[i][x][y][1]) + int(flows[i][x][y][2])	#this is not really a metric
				if(intensity <= 255*2 + 255):#75
					new_flows[i][x][y] = flows[i][x][y]

for i in range(len(flows)):
	cv2.imshow("Pre",cv2.imread(path + "\\" + images[i]))
	cv2.imshow("Post",cv2.imread(path + "\\" + images[i+1]))
	cv2.imshow("OpticalFLow",flows[i])
	hsvImg = cv2.cvtColor(new_flows[i],cv2.COLOR_BGR2HSV)

	#multiple by a factor to change the saturation
	hsvImg[...,1] = hsvImg[...,1]*1.4

	new_flows[i] =cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
	cv2.imshow("OpticalFLowFiltered",new_flows[i])
	cv2.waitKey(0)
