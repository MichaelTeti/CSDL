import tensorflow as tf
import numpy as np
from scipy.misc import *
import os
import glob
from skimage.util import view_as_windows
from scipy.io import loadmat
import functools
from wraps import doublewrap, define_scope
from image_reader import read_ims


def normalize(data):
  return (data-np.mean(data, axis=0))/(np.std(data, axis=0)+1e-6)


def LCA(y, iters, batch_sz, num_dict_features=None, D=None):
  ''' Dynamical systems neural network used for sparse approximation of an
      input vector.

      Args: 
           y: input signal or vector
           num_dict_features: number of dictionary patches to learn.
           iters: number of LCA iterations.
           batch_sz: number of samples to send to the network at each iteration.
           D: The dictionary to be used in the network. 
  '''
  assert(num_dict_features is None or D is None), 'provide D or num_dict_features, not both'
  if num_dict_features is not None:
    D=np.random.randn(y.shape[0], num_dict_features)
  for i in range(iters):
    batch=y[:, np.int32(np.floor(np.random.rand(batch_sz)*y.shape[1]))]
    D=tf.matmul(D, tf.diag(1/(tf.sqrt(tf.reduce_sum(D**2, 0))+1e-6)))
    a=tf.matmul(tf.transpose(D), batch)
    a=tf.matmul(a, tf.diag(1/(tf.sqrt(tf.reduce_sum(a**2, 0))+1e-6)))
    a=0.3*a**3
    D=D+tf.matmul((batch-tf.matmul(D, a)), tf.transpose(a))
  return sess.run(D), sess.run(a)


def visualize_dict(D, d_shape, patch_shape):
  vis_d=np.zeros([d_shape[0]*patch_shape[0], d_shape[1]*patch_shape[1]])
  for row in range(d_shape[0]):
    for col in range(d_shape[1]):
      resized_patch=np.reshape(D[:, row*d_shape[1]+col], [patch_shape[0], patch_shape[1]])
      vis_d[row*patch_shape[0]:row*patch_shape[0]+patch_shape[0], 
            col*patch_shape[1]:col*patch_shape[1]+patch_shape[1]]=resized_patch
  imshow(vis_d)




class ELM(object):
  '''Single, fully-connected hidden layer extreme learning machine.
     
     Args: 
          input_data: training examples with shape n x num_features.
          input_labels: n-dimensional one-hot vector. '''
  
  def __init__(self, data, labels):
    
    self.X=data
    self.Y=labels
    self.batch_sz=batch_sz
    self.prediction
    self.optimize
    self.accuracy

  @define_scope(initializer=tf.contrib.slim.xavier_initializer())
  def prediction(self):
    net=tf.matmul(self.X, tf.random_normal([lfjsdl;jfl;sjdfjsdjsfakj], dtype=tf.float32))
    net=tf.nn.relu(net+tf.Variable(tf.constant(0.1)))
    net=tf.matmul(net, tf.Variable(tf.truncated_normal([200, 1], dtype=tf.float32)))
    return tf.nn.sigmoid(net+tf.Variable(tf.constant(0.1, shape=[1, ])))

  @define_scope
  def optimize(self):
    cost=tf.reduce_sum((self.Y-self.prediction)**2, reduction_indices=0)
    return tf.train.AdamOptimizer(1e-4).minimize(cost)  
 
  @define_scope
  def accuracy(self):
   return tf.reduce_mean(tf.cast(tf.equal(tf.round(self.prediction), self.Y), tf.float32))


imsz=200
ps=12  # size of the images
measurements=300 # number of compressed measurements to take
k=441 # number of patches in dictionary


# read images from file and resize if not saved already
try:
  data=np.load('oxford_flower_NHWC.npy')
  labels=np.load('oxford_flower_labels.npy')
except IOError:
  data, labels=read_ims('/home/mpcr/Documents/MT/CSDL/17flowers/jpg', imsz)

# get patches to learn dictionary from
random=np.int32(np.floor(np.random.rand(70)*data.shape[0]))
patches=view_as_windows(data[random, :, :, :], (1, ps, ps, 3))

# reshape into vectors
patches=np.transpose(patches.reshape([patches.shape[0]*
			              patches.shape[1]*
			              patches.shape[2]*
			              patches.shape[3], -1]))

# normalize data
patches=normalize(patches)

# random matrix for compressive sampling
rd=np.sign(np.random.randn(measurements, 3*ps**2)/10.0)

# take compressed measurements with random matrix
patches=np.matmul(rd, patches)


with tf.Session() as sess:
  
  # learn dictionary
  print('Learning features of the data...')
  dict_, alpha_=LCA(patches, 300, 75, num_dict_features=k)


##############################test new images#######################################

  batch_sz=45

  x=tf.placeholder(dtype=tf.float32, shape=[None, 3*ps**2*(imsz-ps)**2])
  y=tf.placeholder(dtype=tf.float32, shape=[None, ])

  elm=ELM(x, y)
  
  sess.run(tf.global_variables_initializer())
  a=0
  X=np.zeros([batch_sz, 3*ps**2*(imsz-ps)**2])
  for i in range(100000):
    r=np.int32(np.floor(np.random.rand(batch_sz)*data.shape[0]))
    batch=view_as_windows(data[r, :, :, :], (1, ps, ps, 3))
    batch=np.transpose(batch.reshape([batch.shape[0]*
		                      batch.shape[1]*
			 	      batch.shape[2]*
			 	      batch.shape[3], -1]))
    
    batch=LCA(batch, 1, batch_sz, D=dict_)
    for j in range(batch_sz):
      X[j, :]=batch[:, j*(imsz-ps)**2:j*((imsz-ps)**2)+(imsz-ps)**2].flatten()
    sess.run(elm.optimize, {x: X, y: labels[r]})
    acc=sess.run(elm.accuracy, {x: X, y: labels[r]})
    print('Iteration: %d   acc: %f\r'%(i, acc))
    #if i%100 == 0:
    #  print('val_acc: %f'%(sess.run(elm.accuracy, {x: elm_testd, y: elm_testl})))





