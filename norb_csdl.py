from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
from scipy.misc import imshow, imresize
from sklearn.preprocessing import scale
import tflearn
import matplotlib.pyplot as plt
import h5py

m=4608


def weights(shape, varname):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=varname)

def biases(shape, varname):
    return tf.Variable(tf.constant(0.0, shape=shape), name=varname)

def get_batch(data, labels, batch_size):
    rand=np.random.randint(0, data.shape[0], batch_size)
    return data[rand, :], labels[rand, :]


# Data loading and preprocessing
# X=np.loadtxt('mnist_train.csv', delimiter=',')
# Y=np.zeros([X.shape[0], 10])
# for i in range(Y.shape[0]):
#     Y[i, np.int32(X[i, 0])]=1
# X=X[:, 1:]
# n=X.shape[1]
#
# testX=np.loadtxt('mnist_test.csv', delimiter=',')
# testY=np.zeros([testX.shape[0], 10])
# for i in range(testY.shape[0]):
#     testY[i, np.int32(testX[i, 0])]=1
# testX=testX[:, 1:]

Y=h5py.File('norb_small_labels.h5', 'r')
testY=np.asarray(Y['test_labels'])
Y=np.asarray(Y['train_labels'])

X=h5py.File('norb_small.h5', 'r')
testX=np.asarray(X['test'])
X=np.asarray(X['train'])
n=X.shape[1]


# Feature Scaling
X=scale(X, 1)
testX=scale(testX, 1)

#Compressed Sensing
A=np.random.randn(n, m)
X=np.matmul(X, A)
testX=np.matmul(testX, A)

# network
input_layer = tflearn.input_data(shape=[None, m])
dense1 = tflearn.fully_connected(input_layer, 800, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 800, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, 5, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=110, validation_set=(testX, testY),
          show_metric=True, run_id="norb_cs_4608/18432")
