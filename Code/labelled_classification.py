import cv2
import numpy as np
import sys
from sklearn.externals import joblib
from classificationAlgo import getFeature
import json
import glob
fil = sys.argv[1]
algo = sys.argv[2]
font = cv2.FONT_HERSHEY_SIMPLEX
model = {"sift":'classifier1/linearSVC_SIFT_random/filename.pkl', "hog":'classifier1/linearSVC_HOG_random/filename.pkl', "cnn":'classifier1/linearSVC_CNN_random/filename.pkl'}
# model = {"sift":'classifiers/linearSVC_SIFT_random/filename.pkl', "hog":'classifiers/linearSVC_HOG_random/filename.pkl', "cnn":'classifiers/linearSVC_CNN_random/filename.pkl'}
# ret, frame1 = cap.read()
# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[...,1] = 255
# CLF = {}
# for i in model.keys():

clf = joblib.load(model[algo])

def getframes(labelfil):
    frame_dict  = {}
    with open(labelfil) as json_data:
            d = json.load(json_data)
            for item in d:
                label = d[item]["label"]
                for frame in d[item]["boxes"]:
                    if not frame in frame_dict:
                        frame_dict[frame] = []
                    # print frame, label
                    temp_list = [d[item]["boxes"][frame]["occluded"],d[item]["boxes"][frame]["outside"],d[item]["boxes"][frame]["xtl"], d[item]["boxes"][frame]["ytl"], d[item]["boxes"][frame]["xbr"], d[item]["boxes"][frame]["ybr"], label]
                    frame_dict[frame].append(temp_list)

    return frame_dict

label_source_dir = 'data/'
video_source_dir = 'videoData/'
#
# Dir = ['Car/', ]
# vid_filelist = glob.glob(video_source_dir+'*')
vid_filelist = [fil]
print (vid_filelist)
for fil in vid_filelist:
    label_fil = fil[:-4]+'.json'
    label_fil = label_source_dir + label_fil
    cap = cv2.VideoCapture(fil)
    frame_dict = getframes(label_fil)
    print (fil,len(frame_dict))
    count = 0
    im_no = 0
    while(1):

        ret, frame = cap.read()
        if ret == False:
            break
        count += 1
        if(count%2 == 0):
            # print (count)
            key = str(count)
            if count<len(frame_dict):
                if key in frame_dict.keys():
                    for lis in frame_dict[key]:

                        if (lis[1] == 0 and (lis[6] == "Car" or lis[6] == "Motorcycle" or lis[6] == "Person" or lis[6] == "Bicycle" or lis[6] == "Autorickshaw" or lis[6] == "Rickshaw")):
                            (x, y, w, h) = (lis[2],lis[3],lis[4],lis[5])
                            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                            area = (w-x)*(h-y)
                            ## Computing SIFT Points
                            if area > 8000:
                                im  = frame[y:h,x:w]
                                feat_vec = getFeature(im, algo)
                                # print (feat_vec)
                                # clf = CLF[algo]
                                pred = clf.predict(feat_vec)
                                cv2.putText(frame,pred[0],(x,y+20), font, 1,(255,255,255),2,cv2.LINE_AA)
            height = len(frame)
            width = len(frame[0])
            red=2
            frame = cv2.resize(frame, (int(width/red),int(height/red)))
            # rgb2 = cv2.resize(rgb2, (int(width/red),int(height/red)))
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
            # cv2.imwrite('result3/'+fil[15:-5]+str(count)+'.jpg',frame)
