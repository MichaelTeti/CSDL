from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tensorflow import extract_image_patches, reduce_sum, random_normal, reshape

import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = extract_image_patches(network, ksizes=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1], rates=[1, 1, 1, 1],
                                    padding='VALID')
network = reshape(network, [-1, 28, 28, 3, 3, 96])
network = network * random_normal([64, 28, 28, 3, 3, 96])
network = reduce_sum(reshape(network, [-1, 28, 28, 9, 96]), reduction_indices=3)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = extract_image_patches(network, ksizes=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1], rates=[1, 1, 1, 1],
                                    padding='VALID')
network = reshape(network, [-1, 13, 13, 3, 3, 256])
network = network * random_normal([64, 13, 13, 3, 3, 256])
network = reduce_sum(reshape(network, [-1, 13, 13, 9, 256]), reduction_indices=3)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = extract_image_patches(network, ksizes=[1, 3, 3, 1],
                                strides=[1, 2, 2, 1], rates=[1, 1, 1, 1],
                                padding='VALID')
network = reshape(network, [-1, 6, 6, 3, 3, 256])
network = network * random_normal([64, 6, 6, 3, 3, 256])
network = reduce_sum(reshape(network, [-1, 6, 6, 9, 256]), reduction_indices=3)
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
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_flowers_cspooling3x3')
