import matplotlib
import matplotlib.pyplot as plt
import  matplotlib.cm as cm
from matplotlib import pylab
import collections
import cv2

from loadDataSet import LoadDataSet
from loadTest import LoadTest
import numpy as np
from keras.models import  Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD


def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized


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


def convert_output2(outputs):
    return np.eye(len(outputs))


def create_ann():

    ann = Sequential()

    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(7, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(X_train, y_train, nb_epoch=500, batch_size=1, verbose = 0, shuffle=False, show_accuracy = False)

    return ann


def winner(output): # output je vektor sa izlaza neuronske mreze
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

A_tone = LoadDataSet('samples/adur.wav')
B_tone = LoadDataSet('samples/hdur.wav')
C_tone = LoadDataSet('samples/cdur.wav')
D_tone = LoadDataSet('samples/ddur.wav')
E_tone = LoadDataSet('samples/edur.wav')
F_tone = LoadDataSet('samples/fdur.wav')
G_tone = LoadDataSet('samples/gdur.wav')


X_train = []
X_train.append(resize_region(A_tone.DataSet))
X_train.append(resize_region(B_tone.DataSet))
X_train.append(resize_region(C_tone.DataSet))
X_train.append(resize_region(D_tone.DataSet))
X_train.append(resize_region(E_tone.DataSet))
X_train.append(resize_region(F_tone.DataSet))
X_train.append(resize_region(G_tone.DataSet))

x_train = prepare_for_rnn(X_train)
tones = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
y_train = convert_output2(tones)

print "Creating neural network..."
ann = create_ann()
print "Training neural network..."
ann = train_ann(ann, x_train, y_train)

print "Result:"
test_tone = LoadTest('samples/test2.wav')

print len(test_tone.TestSet)

i = 0
j = i + 100

while j < len(test_tone.TestSet):
    test = []
    test.append(resize_region(test_tone.TestSet[i:j]))
    test = prepare_for_rnn(test)
    result = ann.predict(np.array(test, np.float32))
    z = j

    if len(test_tone.TestSet) - j < 100:
        j = len(test_tone.TestSet)
        i = z
    else:
        j += 100
        i = z

    print display_result(result, tones)
