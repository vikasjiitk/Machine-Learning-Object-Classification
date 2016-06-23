import cv2
import numpy as np
import sys
from sklearn.externals import joblib
from classificationAlgo import getFeature
filename = sys.argv[1]
cap = cv2.VideoCapture(filename)
algo = sys.argv[2]
font = cv2.FONT_HERSHEY_SIMPLEX
model = {"sift":'classifier1/linearSVC_SIFT_random/filename.pkl', "hog":'classifier1/linearSVC_HOG_random/filename.pkl', "cnn":'classifier1/linearSVC_CNN_random/filename.pkl'}
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
# CLF = {}
# for i in model.keys():

clf = joblib.load(model[algo])
count  = 0
while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    rgb = cv2.GaussianBlur(rgb, (21, 21), 0)
    kernelo = np.ones((30,30),np.uint8)

    # rgb = cv2.morphologyEx(rgb, cv2.MORPH_OPEN, kernelo)
    rgb2 = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    rgb2 = cv2.GaussianBlur(rgb2, (21, 21), 0)
    rgb2 = cv2.morphologyEx(rgb2, cv2.MORPH_OPEN, kernelo)
    thresh = 4
    rgb2 = cv2.threshold(rgb2, thresh, 255, cv2.THRESH_BINARY)[1]
    # rgb2 = cv2.morphologyEx(rgb2, cv2.MORPH_OPEN, kernelo)
    if count%5==0:
        kerneld = np.ones((20,20),np.uint8)
        rgb2 = cv2.dilate(rgb2, kerneld, iterations = 4)
        (_,cnts, _) = cv2.findContours(rgb2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            marginx = w/20
            marginy = h/20
            x = int(x+marginx)
            y = int(y+marginy)
            w = int(w-2*marginx)
            h = int(h-2*marginy)
            if w*h < 12000:
                continue
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)
            im = frame2[y:y+h, x:x+w]
            feat_vec = getFeature(im, algo)
            print (feat_vec)
            # clf = CLF[algo]
            pred = clf.predict(feat_vec)
            cv2.putText(frame2,pred[0],(x,y+20), font, 1,(255,255,255),2,cv2.LINE_AA)
            # cv2.imshow('im',im)
            # cv2.waitKey(1000)
        height = len(frame2)
        width = len(frame2[0])
        red=2
        frame2 = cv2.resize(frame2, (int(width/red),int(height/red)))
        rgb2 = cv2.resize(rgb2, (int(width/red),int(height/red)))
        rgb3 = cv2.cvtColor(rgb2,cv2.COLOR_GRAY2BGR)
        w1 = w2 = int(width/red)
        h1 = h2 = int(height/red)
        nWidth = w1+w2
        nHeight = max(h1,h2)
        hdif = 0
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
        newimg[hdif:hdif+h2, :w2] = frame2
        newimg[:h1, w2:w1+w2] = rgb3
        # cv2.imshow('frame2',frame2)
        # cv2.waitKey(1)
        # cv2.imwrite('result3/optical/'+filename[15:-5]+str(count)+'.jpg',newimg)
        cv2.imshow('result3/opticaljpg',newimg)
        cv2.waitKey(1)
        # cv2.imwrite('result3/optical/'+filename[15:-5]+str(count)+'.jpg',frame2)
    # cv2.imshow('rgb',rgb2)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    # elif k == ord('s'):
    #     cv2.imshow('frame2',frame2)
    #     cv2.imshow('rgb',rgb)
    count += 1
    prvs = next

cap.release()
cv2.destroyAllWindows()
