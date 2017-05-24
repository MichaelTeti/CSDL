from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.core import *
from tensorflow import extract_image_patches
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale

ks=10

import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
X=X/255.
testX=testX/255.


a=np.random.randn(34, 29, 29, ks**2, 96)
b=np.random.randn(34, 15, 15, ks**2, 256)
c=np.random.randn(34, 8, 8, ks**2, 256)

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = extract_image_patches(network, ksizes=[1, ks, ks, 1],
                     strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding='SAME')
network = reshape(network, [-1, 29, 29, ks**2, 96])
network = tf.reduce_sum(network*a, 3)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = extract_image_patches(network, ksizes=[1, ks, ks, 1],
                     strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding='SAME')
network = reshape(network, [-1, 15, 15, ks**2, 256])
network = tf.reduce_sum(network*b, 3)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = extract_image_patches(network, ksizes=[1, ks, ks, 1],
                     strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding='SAME')
network = reshape(network, [-1, 8, 8, ks**2, 256])
network = tf.reduce_sum(network*c, 3)
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
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=34, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_oxflowers17_cspooling3x3_2')

