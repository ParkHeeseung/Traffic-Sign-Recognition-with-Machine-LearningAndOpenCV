#import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skimage import exposure
from skimage import feature
from imutils import paths

import argparse
import imutils
import numpy as np
import cv2

HSV_RED_LOWER = np.array([0, 100, 100])
HSV_RED_UPPER = np.array([10, 255, 255])
HSV_RED_LOWER1 = np.array([160, 100, 100])
HSV_RED_UPPER1 = np.array([179, 255, 255])

HSV_YELLOW_LOWER = np.array([10, 80, 120])
HSV_YELLOW_UPPER = np.array([40, 255, 255])

HSV_BLUE_LOWER = np.array([80, 160, 65])
HSV_BLUE_UPPER = np.array([140, 255, 255])

#construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
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

	# gray = cv2.resize(gray, (400, 400))

	# kernel = np.ones((1, 1), np.uint8)
	# erosion = cv2.erode(gray, kernel, iterations=1)

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
	logo = cv2.resize(gray, (128, 128))
	# cv2.imshow("logo", logo)

	# extract Histogram of Oriented Gradients from the logo
	H = feature.hog(logo, orientations=8, pixels_per_cell=(12, 12),
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

cap = cv2.VideoCapture(0);

while True:
	ret, img = cap.read();
	cv2.imshow("frame", img)
	# img = cv2.resize(img, (400, 400))

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# cv2.imshow("hsv", hsv)
	redBinary = cv2.inRange(hsv, HSV_RED_LOWER, HSV_RED_UPPER)
	redBinary1 = cv2.inRange(hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
	redBinary = cv2.bitwise_or(redBinary, redBinary1)
	# cv2.imshow("red", redBinary)
	yellowBinary = cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
	# cv2.imshow("yellow", yellowBinary)
	blueBinary = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
	# cv2.imshow("blue", blueBinary)
	# binary = cv2.bitwise_and(cv2.bitwise_or(yellowBinary, blueBinary), cv2.bitwise_not(redBinary))
	binary = cv2.bitwise_or( blueBinary, (redBinary))

	image, contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	print(len(contours))

	for cnt in contours:
		area = cv2.contourArea(cnt)
		# print(area)
		binary = cv2.drawContours(binary, [cnt], -1, (255,255,255), -1)


	image, goodContours, hierachy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	kernel = np.ones((3, 3), np.uint8)

	gray = cv2.bitwise_and(binary, gray)

	# gray = cv2.bitwise_and(cv2.dilate(binary, kernel, iterations = 1), gray)
	# cv2.imshow("hi", gray)


	for cnt in goodContours:
		area = cv2.contourArea(cnt)
		print(area)
		if area > 2000.0 :
			x, y, w, h = cv2.boundingRect(cnt)
			rate = w / h
			if rate > 0.6 and rate < 1.4 :
				cv2.rectangle(img, (x, y), (x+w, y+h), (200, 152, 50), 2)
				inputImage = gray[y:y+h, x:x+w]
				#
				# kernel = np.ones((1, 1), np.uint8)
				# erosion = cv2.erode(inputImage, kernel, iterations=1)
				logo = cv2.resize(inputImage, (128, 128))
				cv2.imshow("logo", logo)
				(H, hogImage) = feature.hog(logo, orientations=8, pixels_per_cell=(12, 12), \
					cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", visualise=True)

				cv2.imshow("hog", hogImage)
				pred = model.predict(H.reshape(1, -1))[0]

				cv2.putText(img, pred.title(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, \
					(70, 255, 0), 2)




	cv2.imshow("candidates", img)
	if cv2.waitKey(1) == 27:
		break;






# 색공간 변환
#
# 이진화
#
# contours
#
# copyTo
#
# ROI
