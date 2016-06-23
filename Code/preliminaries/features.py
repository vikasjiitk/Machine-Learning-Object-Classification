import cv2
import json
import glob
from sklearn.cluster import KMeans
import csv

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

label_source_dir = '.git/data/'
video_source_dir = '.git/videoData/'

Dir = ['Car/', ]
vid_filelist = glob.glob(video_source_dir+'*')
# (vid_filelist) = [video_source_dir+'datasample1.mov']

tr_images = []
tr_labels = []
no_sift = 500
sift = cv2.xfeatures2d.SIFT_create(nfeatures=no_sift, contrastThreshold = 0.01, edgeThreshold = 100, sigma =0.4)
# sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.01, edgeThreshold = 100, sigma =0.4)
sift_tr = []
index = 0
for fil in vid_filelist:
    label_fil = fil[15:-4]+'.json'
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
        if(count%20 == 0):
            # print (count)
            key = str(count)
            if count<len(frame_dict):
                if key in frame_dict.keys():
                    for lis in frame_dict[key]:

                        if (lis[1] == 0 and (lis[6] == "Car" or lis[6] == "Motorcycle" or lis[6] == "Person" or lis[6] == "Bicycle" or lis[6] == "Autorickshaw" or lis[6] == "Rickshaw")):
                            (x, y, w, h) = (lis[2],lis[3],lis[4],lis[5])
                            # cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                            area = (w-x)*(h-y)
                            ## Computing SIFT Points
                            if area > 8000:
                                im  = frame[y:h,x:w]
                                # cv2.imshow('frame',im)
                                # cv2.waitKey(500)
                                gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                                (kps, descs) = sift.detectAndCompute(gray, None)
                                # print (area, len(kps))
                                nos = min(no_sift, len(kps))
                                if (nos > 50):
                                    im_no += 1
                                    cv2.imwrite(label_fil+'1/'+lis[6]+'_'+str(lis[0])+'_'+str(im_no)+'.jpg', im)
                                    print (im_no)
                                    for point in range(nos):
                                        sift_tr.append(descs[point])
                                    tr_images.append([index, index+nos])
                                    index += nos
                                    tr_labels.append([lis[6],lis[0]])

        # red=3
        # height = len(frame)
        # width = len(frame[0])
        # frame = cv2.resize(frame, (int(width/red),int(height/red)))
        # cv2.imshow('frame',frame)
        # cv2.waitKey(10)


    print (count/20)

siftfile = open('siftpoints.csv', 'w')
wr = csv.writer(siftfile, quoting=csv.QUOTE_ALL)
for i in range(len(sift_tr)):
    wr.writerow([sift_tr[i]])

imfile = open('im_index_points.csv', 'w')
wr = csv.writer(imfile, quoting=csv.QUOTE_ALL)
for i in range(len(tr_images)):
    wr.writerow([tr_images[i]])

imlfile = open('im_label_points.csv', 'w')
wr = csv.writer(imlfile, quoting=csv.QUOTE_ALL)
for i in range(len(tr_labels)):
    wr.writerow([tr_labels[i]])

# no_clus = 500
# kmeans = KMeans(init='k-means++', n_clusters=no_clus, n_init=10)
# y_c = kmeans.fit_predict(sift_tr)
# print (y_c)
# print (len(sift_tr), len(sift_tr[0]))
