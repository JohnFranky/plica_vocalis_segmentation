import cv2

path = r"E:\Eigene Dateien Jonathan\studium\6. Semester\Bachlor\videos\heatmaps\glottis"

img = cv2.imread(path,0)

for x in range(0,len(img)):
    for y in range(0,len(img[0])):
        if img[x][y] == 1:
            img[x][y] = 80
        elif img[x][y] == 2:
            img[x][y] = 160
        elif img[x][y] == 3:
            img[x][y] = 256

cv2.imshow("img", img)
cv2.waitKey(0)