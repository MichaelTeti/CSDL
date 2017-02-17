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
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
from scipy.misc import imshow
from tflearn.data_augmentation import ImageAugmentation

import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(150, 150))
X=X.reshape([X.shape[0], -1])
X=(X-np.mean(X, axis=0))/(np.std(X, axis=0)+1e-6)
X=X.reshape([X.shape[0], 150, 150, 3])
img_aug = ImageAugmentation()
img_aug.add_random_rotation(max_angle=33.)
#X=np.mean(X, axis=3)
#X=np.transpose(X.reshape([X.shape[0], -1]))
#X=(X-np.mean(X, axis=0))/(np.std(X, axis=0)+1e-6)
#m=100
#D=np.sign(np.random.randn(m, X.shape[0]))
#X=np.transpose(np.matmul(D, X))
#X=X.reshape([X.shape[0], 10, 10, 1])

# Building 'AlexNet'
network = input_data(shape=[None, 150, 150, 3], data_augmentation=img_aug)
network = conv_2d(network, 96, 11, strides=4, weights_init='normal', trainable=False)
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, weights_init='normal', trainable=False)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=3)
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
snapshot_epoch=False, run_id='cifar10_altconvrand2_size170')
