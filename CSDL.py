# -*- coding: utf-8 -*-

""" AlexNet.
Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import

import tflearn
from scipy.io import loadmat
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import os
import sys
from skimage.util import view_as_windows as vaw
from scipy.misc import *
import matplotlib.pyplot as plt
from tflearn.data_augmentation import ImageAugmentation

ps=5
add=np.int32((ps-1)/2)
measurements=100
epsilon=1e-6

data=loadmat('Pavia.mat')
labels=loadmat('Pavia_gt.mat')
data=data['pavia']
labels=labels['pavia_gt']


data=np.pad(data, ((add, add), (add, add), (0, 0)), 'edge')
r, c=np.nonzero(labels)
#data=data[r, c, :]
#rand=np.int32(np.random.rand(4)*data.shape[1])
#data=data[:, rand]
#Y=np.zeros([data.shape[0], np.amax(labels)+1])
#labels=labels[r, c]
#for i in range(Y.shape[0]):
#  Y[i, labels[i]]=1
#X=data[:, np.newaxis, np.newaxis, :]
r=r+add
c=c+add
X=np.zeros([r.size, 19, 19, 1])
Y=np.zeros([r.size, np.amax(labels)+1])
rand=np.int32(np.random.rand(361)*(102*ps**2))

for i in range(r.size):
  dflat=data[r[i]-(add):r[i]+(add+1), c[i]-add:c[i]+(add+1), :].reshape([1, 102*ps**2])
  X[i, :, :, :]=dflat[:, rand].reshape([19, 19, 1])
  Y[i, labels[r[i]-add, c[i]-add]]=1

X=(X-np.mean(X, axis=0))/(np.std(X, axis=0)+epsilon)

# Building 'AlexNet'
network = input_data(shape=[None, 19, 19, 1])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
snapshot_epoch=False, run_id='Pavia_5x5patch19x19random361')
