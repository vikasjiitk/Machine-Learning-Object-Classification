import csv
import time
from sklearn.externals import joblib
from tqdm import tqdm

images = []
f = open('im_label_points.csv', 'r')
data = csv.reader(f, delimiter = ' ')
for row in data:
    images.append(eval(row[0]))
f.close()
f = open('im_label_points2.csv', 'r')
data = csv.reader(f, delimiter = ' ')
for row in data:
    images.append(eval(row[0]))
f.close()
print (len(images))

image_index = []
f = open(d+'im_index_points.csv', 'r')
data = csv.reader(f, delimiter = ' ')
for row in data:
    image_index.append(eval(row[0]))
f.close()
f = open('im_index_points2.csv', 'r')
data = csv.reader(f, delimiter = ' ')
for row in data:
    image_index.append(eval(row[0]))
f.close()
print (len(image_index))

print('loading sift points')
siftpoints = []
f = open('siftpoints.csv', 'r')
data = csv.reader(f, delimiter = ' ')
for row in tqdm(data):
    desc = ",".join(row[0].split())
    desc = desc[:1] + desc[2:]
    siftpoints.append(eval(desc))
f.close()
f = open('siftpoints2.csv', 'r')
data = csv.reader(f, delimiter = ' ')
for row in data:
    desc = ",".join(row[0].split())
    desc = desc[:1] + desc[2:]
    siftpoints.append(eval(desc))
f.close()
print (len(siftpoints))
print ('loaded')
from sklearn.cluster import MiniBatchKMeans, KMeans

# no_sift_list = [500, 400, 300, 200, 100, 50]
no_sift_list = [300]
for no_sift in no_sift_list:
    X = []
    y = []
    for i in range(len(image_index)):
        no = min(no_sift,image_index[i][1]-image_index[i][0])
        # print (len(X), image_index[i][0], no)
        X.extend(siftpoints[image_index[i][0]:image_index[i][0]+no])
        y.append(images[i][0])


    # num_clusters_list = [500, 600, 700, 800, 900, 1000]
    num_clusters_list = [500]
    for num_clusters in num_clusters_list:
        print ("Training")
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=num_clusters, batch_size = 100, n_init = 5)
        t0 = time.time()
        mbk.fit(X)
        print ("dumping")
        joblib.dump(mbk, 'models/M'+str(no_sift)+'_'+str(num_clusters)+'.pkl')
        t_mini_batch = time.time() - t0
        print (no_sift, num_clusters, t_mini_batch)
        mbk_means_labels = mbk.labels_
        feat_vector = [[0 for i in range(num_clusters)] for j in range(len(image_index))]
        pos=0
        print ("Computing features")
        for i in tqdm(range(len(image_index))):
            no = min(no_sift,image_index[i][1]-image_index[i][0])
            for j in range(no):
                feat_vector[i][mbk_means_labels[pos]] += 1
                pos +=1
        imlfile = open('features2/features'+str(no_sift)+'_'+str(num_clusters)+'.csv', 'w')
        wr = csv.writer(imlfile, quoting=csv.QUOTE_ALL)
        print ("Saving features in file")
        for i in tqdm(range(len(feat_vector))):
            temp = []
            for ele in feat_vector[i]:
                temp.append(ele)
            wr.writerow(temp)
