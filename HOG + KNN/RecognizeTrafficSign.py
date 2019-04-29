#import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skimage import exposure
from skimage import feature
from imutils import paths

import argparse
import imutils
import cv2
import numpy as np


#construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())



# initialize the data matrix and labels
print ("[INFO] extracting features...")
data = []
labels = []

# loop over the image paths in the training set
for imagePath in paths.list_images(args["training"]):
	# extract the make of the car
	make = imagePath.split("/")[-2]

	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gray = cv2.resize(gray, (400, 400))

	kernel = np.ones((5, 5), np.uint8)
	erosion = cv2.erode(gray, kernel, iterations=1)

	# edged = imutils.auto_canny(gray)


	# find contours in the edge map, keeping only the largest one which
	# is presmumed to be the car logo
	# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	# 	cv2.CHAIN_APPROX_SIMPLE)
	# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# c = max(cnts, key=cv2.contourArea)
	#
	# # extract the logo of the car and resize it to a canonical width
	# # and height
	# (x, y, w, h) = cv2.boundingRect(c)
	# logo = gray[y:y + h, x:x + w]
	logo = cv2.resize(erosion, (48, 48))
	# cv2.imshow("logo", logo)
	cv2.waitKey(0)

	# extract Histogram of Oriented Gradients from the logo
	H = feature.hog(logo, orientations=8, pixels_per_cell=(8, 8),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")

	# update the data and labels
	data.append(H)
	labels.append(make)

# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
model = KNeighborsClassifier(n_neighbors=1)
# model = SVC(gamma = 'guto')

model.fit(data, labels)
print(model)
print("[INFO] evaluating...")

# loop over the test dataset
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
	# load the test image, convert it to grayscale, and resize it to
	# the canonical size
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	kernel = np.ones((5, 5), np.uint8)
	erosion = cv2.erode(gray, kernel, iterations=1)


	logo = cv2.resize(erosion, (48, 48))
	# cv2.imshow("reuslt", logo)
	# logo = logo[50:350, 50:350]

	# extract Histogram of Oriented Gradients from the test image and
	# predict the make of the car
	(H, hogImage) = feature.hog(logo, orientations=8, pixels_per_cell=(8, 8),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", visualise=True)
	pred = model.predict(H.reshape(1, -1))[0]
	pred1 = model.predict_proba(H.reshape(1, -1))[0]
	print(pred1)
	#print(dir(pred))
	# pred = model.predict(H)


	# visualize the HOG image
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	hogImage = hogImage.astype("uint8")
	# cv2.imshow("HOG Image #{}".format(i + 1), hogImage)

	# draw the prediction on the test image and display it
	cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 3)
	cv2.imshow("Test Image #{}".format(i + 1), image)
	cv2.waitKey(0)
