import cv2
import numpy as np
from skimage.feature import hog
import csv
import glob
import random
from tqdm import tqdm
Dir = ['bicycle/', 'Car/', 'motorcycle/', 'person/', 'rickshaw/', 'autorickshaw/'  ]
# Dir = ['temp2/', 'temp/'  ]
features = []
labels = []

for d in Dir:
    im_filelist = glob.glob(d+'*')
    no_im = len(im_filelist)
    t = min(no_im, 400)
    print (d,t)
    im_pos = random.sample(range(0, no_im), t)
    for i in tqdm(im_pos):
        fil = im_filelist[i]
        img = cv2.resize(cv2.imread(fil), (128, 128)).astype(np.float32)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        features.append(hog(gray))
        print(len(features[0]))
        labels.append(d[:-1])

imfile = open('allfeatures/featureshog.csv', 'w')
imlfile = open('allfeatures/labelhog.csv', 'w')
wr = csv.writer(imfile, quoting=csv.QUOTE_ALL)
wr2 = csv.writer(imlfile, quoting=csv.QUOTE_ALL)
print ("Saving features in file")
for i in tqdm(range(len(features))):
    temp = []
    for ele in features[i]:
        temp.append(ele)
    wr.writerow(temp)
    wr2.writerow([labels[i]])

# features = []
# im_labels = []
# # Dir = ['temp2/', 'temp/']
# Dir = ['bicycle/', 'Car/', 'motorcycle/', 'person/', 'rickshaw/', 'autorickshaw/'  ]
#
# im_filelist = []
# print ('Reading Images...')
# for sourceDir in sourceDirs:
#     im_file = glob.glob(sourceDir+'*')
#     im_filelist.extend(im_file)
#
# hogl = []
# print ('Sorting Images...')
# im_filelist.sort()
# print ('Sorting done.')
# count = 0
# for fil in tqdm(im_filelist):
#     tear = (fil.split('/'))
#     name = tear[3].split("_")
#     im_labels.append(name[0])
#     print ('Image number:', count, ', Name:', fil)
#     count += 1
#     im = cv2.resize(cv2.imread(fil), (128, 128)).astype(np.float32)
#     gray_img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#     hogl.append(hog(gray_img))
# # print (count)
# # print (len(hogl), len(hogl[0]))
