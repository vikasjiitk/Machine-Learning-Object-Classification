import csv
import time
from sklearn.externals import joblib
from tqdm import tqdm
import random
from sklearn.cluster import MiniBatchKMeans, KMeans
import warnings
warnings.filterwarnings("ignore")


# Dir = ['temp2/', 'temp/']
Dir = ['bicycle/', 'Car/', 'motorcycle/', 'person/', 'rickshaw/', 'autorickshaw/'  ]
labels = []
index = []
siftpoints = []

for d in Dir:

    image_index = []
    f = open(d+'im_index_points.csv', 'r')
    data = csv.reader(f, delimiter = ',')
    ind = -1
    for row in data:
        ind += 1
        image_index.append([])
        for i in row:
            image_index[ind].append(eval(i))
    f.close()

    sift = []
    f = open(d+'siftpoints.csv', 'r')
    data = csv.reader(f, delimiter = ',')
    ind = -1
    for row in data:
        ind += 1
        sift.append([])
        for i in row:
            sift[ind].append(eval(i))
    f.close()

    no_im = len(image_index)
    t = min(no_im, 400)
    im_pos = random.sample(range(0, no_im), t)
    print (d, t)
    for i in im_pos:
        labels.append(d[:-1])
        index.append(image_index[i])
        siftpoints.extend(sift[image_index[i][0]:image_index[i][1]])

print ("Training")
num_clusters = 700
mbk = MiniBatchKMeans(init='k-means++', n_clusters=num_clusters, batch_size = 100, n_init = 5)
t0 = time.time()
mbk.fit(siftpoints)
print ("dumping")
joblib.dump(mbk, 'models/M_'+str(num_clusters)+'.pkl')
t_mini_batch = time.time() - t0
print (num_clusters, t_mini_batch)
mbk_means_labels = mbk.labels_
feat_vector = [[0 for i in range(num_clusters)] for j in range(len(index))]
pos=0
print ("Computing features")
for i in tqdm(range(len(index))):
    # no = min(no_sift,index[i][1]-image_index[i][0])
    for j in range(index[i][0],index[i][1]):
        feat_vector[i][mbk_means_labels[pos]] += 1
        pos +=1
imfile = open('features_new_SIFT/featuresf_'+str(num_clusters)+'.csv', 'w')
imlfile = open('features_new_SIFT/labelsf_'+str(num_clusters)+'.csv', 'w')
wr = csv.writer(imfile, quoting=csv.QUOTE_ALL)
wr2 = csv.writer(imlfile, quoting=csv.QUOTE_ALL)
print ("Saving features in file")
for i in tqdm(range(len(feat_vector))):
    temp = []
    for ele in feat_vector[i]:
        temp.append(ele)
    wr.writerow(temp)
    wr2.writerow([labels[i]])
