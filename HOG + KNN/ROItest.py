import numpy as np
import cv2


img = cv2.imread("dd.png");
cv2.imshow("frame", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# dd = gray[0:0, 50:50]
print(gray[0:1, 3:6])

cv2.waitKey(0)
