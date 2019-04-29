import numpy as np
import cv2

HSV_RED_LOWER = np.array([0, 100, 100])
HSV_RED_UPPER = np.array([10, 255, 255])
HSV_RED_LOWER1 = np.array([160, 100, 100])
HSV_RED_UPPER1 = np.array([179, 255, 255])

HSV_YELLOW_LOWER = np.array([0, 120, 80])
HSV_YELLOW_UPPER = np.array([40, 255, 255])

HSV_BLUE_LOWER = np.array([80, 160, 65])
HSV_BLUE_UPPER = np.array([140, 255, 180])

img = cv2.imread("/home/suki/바탕화면/Traffic Sign Recognition/image/선택 영역_049.png")
cv2.imshow("frame", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)

redBinary = cv2.inRange(hsv, HSV_RED_LOWER, HSV_RED_UPPER)
redBinary1 = cv2.inRange(hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
redBinary = cv2.bitwise_or(redBinary, redBinary1)
yellowBinary = cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
cv2.imshow("yellow", yellowBinary)
blueBinary = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
cv2.imshow("blue", blueBinary)


binary = cv2.bitwise_and(cv2.bitwise_or(yellowBinary, blueBinary), cv2.bitwise_not(redBinary))
cv2.imshow("binary", binary)

image, contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))

for cnt in contours:
    area = cv2.contourArea(cnt)
    print(area)
    binary = cv2.drawContours(binary, [cnt], -1, (255,255,255), -1)

image, goodContours, hierachy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in goodContours:
    area = cv2.contourArea(cnt)
    print(area)
    if area > 200.0 :
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (150, 255, 0), 2)
        # inputData.append(gray[x:y, x+w:y+h])


cv2.imshow("candidates", img)




cv2.waitKey(0)




# 색공간 변환
#
# 이진화
#
# contours
#
# copyTo
#
# ROI
