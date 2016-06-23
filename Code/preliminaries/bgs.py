import sys
import cv2
import numpy as np

filename = sys.argv[1]

kernelo = np.ones((10,10),np.uint8)
kernelc = np.ones((5,5),np.uint8)
kerneld = np.ones((6,6),np.uint8)

cap = cv2.VideoCapture(filename)
red = 3
history = 500

fgbg = cv2.createBackgroundSubtractorMOG2(history = history, varThreshold=16, detectShadows = False)
fgbg.setShadowValue(0)
frameNo = 1
while(1):
    ret, frame = cap.read()
    if ret == False:
        break
    height = len(frame)
    width = len(frame[0])
    frame = cv2.GaussianBlur(frame, (21, 21), 0)
    fgmask = fgbg.apply(frame, learningRate = 10.0/history)
    if(frameNo%1 == 0):
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernelo)
        fgmask = cv2.dilate(fgmask, kerneld, iterations = 4)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernelc)
        # fgmask = cv2.resize(fgmask, (width/2,height/2))
        (_,cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    	# loop over the contours
        for c in cnts:
            if cv2.contourArea(c) < 4000:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(fgmask, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = "Occupied"
        fgmask = cv2.resize(fgmask, (int(width/red),int(height/red)))
        frame = cv2.resize(frame, (int(width/red),int(height/red)))
        cv2.imshow('frame',fgmask)
        # cv2.waitKey(1000)
        # cv2.imshow('frame',fgmask)
        cv2.waitKey(10)

    frameNo += 1
cap.release()
cv2.destroyAllWindows()
