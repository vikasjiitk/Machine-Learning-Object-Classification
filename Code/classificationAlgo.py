from skimage.feature import hog
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from tqdm import tqdm
import random
from sklearn.externals import joblib

no_sift = 500
sift = cv2.xfeatures2d.SIFT_create(nfeatures=no_sift, contrastThreshold = 0.01, edgeThreshold = 100, sigma =0.4)
mbk = joblib.load('models/M_700.pkl')
def getFeature(img, algo):
    if (algo == 'hog'):
        img = cv2.resize(img, (128, 128)).astype(np.float32)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        feature = hog(gray)
        return feature
    elif (algo == 'cnn'):
        global model2
        im = cv2.resize(img, (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        feature = model2.predict(im)
        return feature[0]
    elif (algo == 'sift'):
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        (kps, descs) = sift.detectAndCompute(gray, None)
        no = len(kps)
        num_clusters = 700
        y = mbk.predict(descs)
        feature = [0 for i in range(num_clusters)]
        for i in y:
            feature[i]+=1
        return feature

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

model = VGG_16('.git/vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy')
model2 = Sequential()
model2.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model2.add(Convolution2D(64, 3, 3, activation='relu', weights=model.layers[1].get_weights()))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(64, 3, 3, activation='relu', weights=model.layers[3].get_weights()))
model2.add(MaxPooling2D((2,2), strides=(2,2)))

model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(128, 3, 3, activation='relu', weights=model.layers[6].get_weights()))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(128, 3, 3, activation='relu', weights=model.layers[8].get_weights()))
model2.add(MaxPooling2D((2,2), strides=(2,2)))

model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(256, 3, 3, activation='relu', weights=model.layers[11].get_weights()))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(256, 3, 3, activation='relu', weights=model.layers[13].get_weights()))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(256, 3, 3, activation='relu', weights=model.layers[15].get_weights()))
model2.add(MaxPooling2D((2,2), strides=(2,2)))

model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[18].get_weights()))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[20].get_weights()))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[22].get_weights()))
model2.add(MaxPooling2D((2,2), strides=(2,2)))

model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[25].get_weights()))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[27].get_weights()))
model2.add(ZeroPadding2D((1,1)))
model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[29].get_weights()))
model2.add(MaxPooling2D((2,2), strides=(2,2)))

model2.add(Flatten())
model2.add(Dense(4096, activation='relu', weights=model.layers[32].get_weights()))
model2.add(Dropout(0.5))
model2.add(Dense(4096, activation='relu', weights=model.layers[34].get_weights()))
model2.add(Dropout(0.5))

model2.compile(optimizer=sgd, loss='categorical_crossentropy')
