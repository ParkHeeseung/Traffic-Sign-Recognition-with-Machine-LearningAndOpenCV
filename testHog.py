import cv2

import matplotlib.pyplot as plt
import numpy as np

capture = cv2.VideoCapture(1)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)


while True:

    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, dsize=(128, 128), interpolation=cv2.INTER_AREA)



from skimage import exposure
from skimage import feature
import cv2

(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(8, 8),
	cells_per_block=(2, 2), transform_sqrt=True, , block_norm="L1",
	visualise=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

cv2.imshow("HOG Image", hogImage)
    if cv2.waitKey(1) > 0: break


capture.release()
cv2.destroyAllWindows()
