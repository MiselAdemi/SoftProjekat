import matplotlib
import matplotlib.pyplot as plt
import  matplotlib.cm as cm
from matplotlib import pylab
import collections

from loadDataSet import LoadDataSet
import numpy as np
from keras.models import  Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import cv2


def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_dic = {}
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 100 and h < 100 and h > 15 and w > 20:
            region = image_bin[y:y+h+1,x:x+w+1];
            regions_dic[x] = resize_region(region)
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)

    sorted_regions_dic = collections.OrderedDict(sorted(regions_dic.items()))
    sorted_regions = sorted_regions_dic.values()

    return image_orig, sorted_regions


def matrix_to_vector(m):
    return m.flatten()


def prepare_for_rnn(tones):
    ready_for_rnn = []
    for tone in tones:
        ready_for_rnn.append(matrix_to_vector(tone))

    return ready_for_rnn


def convert_output(seq):
    nn_outputs = []
    for index in range(len(seq)):
        output = np.zeros(len(seq))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

tone = LoadDataSet('samples/ddur.wav')

X_train = tone.DataSet

x_train = prepare_for_rnn(X_train)
tones = ['D']
y_train = convert_output(tones)

model = Sequential()

model.add(Dense(128, input_dim=1, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x_train, y_train, nb_epoch=2000, batch_size=1, verbose=0, shuffle=False, show_accuracy=False)
score = model.evaluate(x_train, y_train, batch_size=16)
