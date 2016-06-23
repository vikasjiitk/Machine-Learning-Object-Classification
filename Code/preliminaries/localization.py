import argparse
import time
import cv2
import imutils
import sys
from sklearn.externals import joblib
from classificationAlgo import getFeature
# filename = sys.argv[1]
algo = 'hog'
font = cv2.FONT_HERSHEY_SIMPLEX
# cap = cv2.VideoCapture(filename)
clf = joblib.load('classifier1/linearSVC_HOG_random/filename.pkl')

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
cap = cv2.VideoCapture(args["image"])
# load the image and define the window width and height
while(1):
	ret, image = cap.read()
	if ret == False:
		break
	# image = cv2.imread(args["image"])
	red = 2
	height = len(image)
	width = len(image[0])
	image = cv2.resize(image, (int(width/red),int(height/red)))
	(winW, winH) = (50, 128)

	for resized in pyramid(image, scale=1.5):
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			im = resized[y:y+winH, x:x+winW]
			cv2.imshow("Window2", im)
			cv2.waitKey(1)
			feat_vec = getFeature(im, algo)
			pred = clf.predict(feat_vec)
			score = clf.decision_function(feat_vec)
			score = max(score[0])
			if(abs(score)>0.8):
				cv2.putText(resized,pred[0],(x,y+20), font, 0.5,(255,255,255),2,cv2.LINE_AA)
			# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
			# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
			# WINDOW

			# since we do not have a classifier, we'll just draw the window
			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			time.sleep(0.025)
