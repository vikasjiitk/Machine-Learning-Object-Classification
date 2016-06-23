import cv2
import json
import glob
from sklearn.cluster import KMeans
import csv
from tqdm import tqdm

# Dir = ['bicycle/', 'Car/', 'motorcycle/', 'person/', 'rickshaw/', 'autorickshaw/'  ]
Dir = ['temp2/', 'temp/'  ]
tr_images = []
tr_labels = []
no_sift = 500
sift = cv2.xfeatures2d.SIFT_create(nfeatures=no_sift, contrastThreshold = 0.01, edgeThreshold = 100, sigma =0.4)

for d in Dir:
    print (d)
    sift_tr = []
    tr_images = []
    im_filelist = glob.glob(d+'*')
    index = 0
    im_no = 0
    for fil in tqdm(im_filelist):
        im = cv2.imread(fil)
        gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        (kps, descs) = sift.detectAndCompute(gray, None)
        nos = min(no_sift, len(kps))
        if (nos > 50):
            im_no += 1
            for point in range(nos):
                sift_tr.append(descs[point])
            tr_images.append([index, index+nos])
            index += nos
    siftfile = open(d+'siftpoints.csv', 'w')
    wr = csv.writer(siftfile, quoting=csv.QUOTE_ALL)
    for i in range(len(sift_tr)):
        temp = []
        for j in sift_tr[i]:
            temp.append(j)
        wr.writerow(temp)

    imfile = open(d+'im_index_points.csv', 'w')
    wr = csv.writer(imfile, quoting=csv.QUOTE_ALL)
    for i in range(len(tr_images)):
        temp = []
        for j in tr_images[i]:
            temp.append(j)
        wr.writerow(temp)
    print (im_no)
