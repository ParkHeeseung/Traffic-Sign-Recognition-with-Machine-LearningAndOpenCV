#Hog Test with Python

from skimage import exposure
from skimage import feature
import cv2


frame = cv2.imread("static.png", cv2.IMREAD_GRAYSCALE)

(H, hogImage) = feature.hog(frame, orientations=9, pixels_per_cell=(8, 8),
	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
	visualise=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("result", hogImage)
cv2.waitKey(0)
