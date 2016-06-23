from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np, glob, csv
from tqdm import tqdm
import random

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

if __name__ == "__main__":

    # Test pretrained model
    model = VGG_16('.git/vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
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

    # out = model2.predict(im)
    # for i in out[0]:
    #     print (i)
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
            # print ('Image number:', count, ', Name:', fil)
            im = cv2.resize(cv2.imread(fil), (224, 224)).astype(np.float32)
            im[:,:,0] -= 103.939
            im[:,:,1] -= 116.779
            im[:,:,2] -= 123.68
            im = im.transpose((2,0,1))
            im = np.expand_dims(im, axis=0)
            out = model2.predict(im)
            # print (len(out[0]))
            # print (out[0])
            features.append(out[0])
            labels.append(d[:-1])

    imfile = open('allfeatures/featurescnn.csv', 'w')
    imlfile = open('allfeatures/labelcnn.csv', 'w')
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
    # # sourceDirs = ['.git/data/datasample1.json1/', '.git/data/dec21h1330.json1/', 'input_video_sample1.json1/', 'input_video_sample2.json1/', 'input_video_sample3.json1/', 'nov92015-1.json1/', 'nov92015-2.json1/']
    # sourceDirs = ['.git/data/input_video_sample2.json1/']# '.git/data/dec21h1330.json1/', 'input_video_sample1.json1/', 'input_video_sample2.json1/', 'input_video_sample3.json1/', 'nov92015-1.json1/', 'nov92015-2.json1/']
    #
    # for sourceDir in sourceDirs:
    #     print (sourceDir)
    #     first = len(sourceDir)
    #     print ('Reading Images...')
    #     im_filelist = glob.glob(sourceDir+'*')
    #     print ('Sorting Images...')
    #     im_filelist.sort()
    #     print ('Sorting done.')
    #     count = 1
    #     for fil in tqdm(im_filelist):
    #         im_labels.append(fil[first:].split('_'))
    #         print ('Image number:', count, ', Name:', fil)
    #         count += 1
    #         im = cv2.resize(cv2.imread(fil), (224, 224)).astype(np.float32)
    #         im[:,:,0] -= 103.939
    #         im[:,:,1] -= 116.779
    #         im[:,:,2] -= 123.68
    #         im = im.transpose((2,0,1))
    #         im = np.expand_dims(im, axis=0)
    #         out = model2.predict(im)
    #         # print (len(out[0]))
    #         # print (out[0])
    #         features.append(out[0])
    # print ('Writing features in csv file')
    # imlfile = open('features_cnn.csv', 'w')
    # imlabel = open('labels_cnn.csv','w')
    # wr = csv.writer(imlfile, quoting=csv.QUOTE_ALL)
    # wr2 = csv.writer(imlabel, quoting=csv.QUOTE_ALL)
    # for i in range(len(features)):
    #     temp=[]
    #     print (im_labels[i])
    #     wr2.writerow(im_labels[i])
    #     for x in features[i]:
    #         temp.append(x)
    #     wr.writerow(temp)
    # # print (np.argmax(out))
