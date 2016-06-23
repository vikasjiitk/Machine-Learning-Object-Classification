from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.datasets import fetch_mldata
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from skimage.feature import hog
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import csv
import os
import pickle
import time
from sklearn.externals import joblib
# t0 = time.time()
import random



i=0
labels = []
val=[]
# for old features
# f = open('features_new_SIFT/labels.csv', 'r')



# for sift points
# f = open('allfeatures/labelssift.csv', 'r')

# for cnn points
# f = open('allfeatures/labelcnn.csv', 'r')

#for hog points
f = open('allfeatures/labelhog.csv', 'r')



data = csv.reader(f, delimiter = ' ')
for row in data:
	# for old dataset which had "bicycle",0 type entries
	# temp = row[0].split(",")
	#labels.append(eval(row[0]))
	# labels.append([temp[0], eval(eval(temp[1]))])
	labels.append(row[0])
f.close()
# print (labels)
labelDict={
	'autorickshaw':0,
	'bicycle':0,
	'Car':0,
	'motorcycle':0,
	'person':0,
	'rickshaw':160
}
images = []
imagesTest =[]
valTest =[]
imagesTrain = []
population=range(0,len(labels))
randomList=random.sample(population,len(labels))
# print randomList
# for old features
# f = open('features_new_SIFT/features300_500.csv', 'r')


# f = open('allfeatures/featuressift.csv', 'r')

# f = open('allfeatures/featurescnn.csv', 'r')

f = open('allfeatures/featureshog.csv', 'r')
data = csv.reader(f, delimiter = ',')
ind = -1
for row in data:
	ind += 1
	images.append([])
	for i in row:
		images[ind].append(eval(i))
# innerTrain=-1
# innerTest=-1
# for row in data:
# 	ind += 1
# 	if labelDict[labels[ind]]<300:# and labels[ind]!='rickshaw':
# 		labelDict[labels[ind]]+=1
# 		innerTrain+=1
# 		images.append([])
# 		val.append(labels[ind])
# 		for i in row:
# 			images[innerTrain].append(eval(i))
# 	elif labelDict[labels[ind]]<400:# and labels[ind]!='rickshaw':
# 		labelDict[labels[ind]]+=1
# 		innerTest+=1
# 		imagesTest.append([])
# 		valTest.append(labels[ind])
# 		for i in row:
# 			imagesTest[innerTest].append(eval(i))
f.close()
for inde in randomList:
	print (inde , labels[inde], labelDict[labels[inde]])
	if labelDict[labels[inde]]<300:
		labelDict[labels[inde]]+=1
		imagesTrain.append(images[inde])
		val.append(labels[inde])
	elif labelDict[labels[inde]]<400:
		labelDict[labels[inde]]+=1
		imagesTest.append(images[inde])
		valTest.append(labels[inde])

# print val
print (labelDict)
print (len(val))
print (len(images))
print (len(imagesTrain))
print (len(labels))
print (len(valTest))
print (len(imagesTest))



"""
count=0
weight=[]
feature1=[]
label1=[]

#print count


image_index = []
f = open('im_index_points.csv', 'r')
data = csv.reader(f, delimiter = ' ')
for row in data:
    image_index.append(eval(row[0]))
f.close()

feature_vec = []
f = open('features/features500_1000.csv', 'r')
data = csv.reader(f, delimiter = ' ')
for row in data:
    feature_vec.append(eval(row[0]))
f.close()
for i in range(0,len(occluded)):
	if(occluded[i]==0):
		count+=1;weight.append(0.7)
		label1.append(labels[i])
		feature1.append(feature_vec[i])
	else: weight.append(1.0)
"""
# clf = RandomForestClassifier(n_estimators=200, max_depth=None,min_samples_split=1, random_state=0)
# clf = DecisionTreeClassifier()
# clf = LinearSVC()
clf=SVC()#intercept_scaling = weight)
clf = clf.fit(imagesTrain,val)
# joblib.dump(clf,'classifiers/SVC_SIFT_random/filename.pkl')
# clf=joblib.load('classifiers/linearSVC_SIFT/filename.pkl')
# joblib.dump(clf,'classifier1/linearSVC_HOG_random/filename.pkl')
ans = clf.predict(imagesTest)
output=clf.classes_
"""
correct=0.0
val=0.0
#print ans.shape
output=clf.classes_
output=[ 'Car','Motorcycle','Person', 'Rickshaw','Bicycle']
print output
#	var.append(c[maxim])
#print labels[:650]

for i in range(0,1000):
	#if(occluded[i]==1):
	#	val+=1;print 'hey'
		if(labels[i]==ans[i]): correct+=1;print 'yup'
	else:
		val+=1.0
		if(labels[i]==ans[i]): correct+=1.0
print correct/val
#confusion= confusion
#print classification_report(labels=)
"""
confusion=confusion_matrix(valTest, ans, labels=output)
print (output)
print (confusion)
print (1.0*confusion.trace()/(1*len(ans)))
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(valTest, ans)
print (accuracy)
# t1 = time.time()
# print t1
# print ans
