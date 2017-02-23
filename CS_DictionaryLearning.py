import tensorflow as tf
import numpy as np
from scipy.misc import *
import os
import glob
from skimage.util import view_as_windows
from scipy.io import loadmat


def read_ims(directory, imsz):
  d=os.getcwd()
  os.chdir(directory)
  num_ims=len(glob.glob1(os.getcwd(), '*'))
  imgs=np.zeros([num_ims, imsz, imsz, 3])
  im_num=0
  for filename in os.listdir(os.getcwd()):
    print(filename)
    im=imresize(imread(filename), [imsz, imsz])
    imgs[im_num, :, :, :]=im
    im_num+=1
  os.chdir(d)
  return imgs


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
  if num_dict_features is not None and D is None:
    D=np.random.randn(y.shape[0], num_dict_features)
  for i in range(iters):
    batch=y[:, np.uint8(np.floor(np.random.rand(batch_sz)*y.shape[1]))]
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


ps=[36, 18]  # size of the images
measurements=300
k=400 # number of patches in dictionary

# read images from file and resize
data=loadmat('ped_images.mat')
absent=data['C1']
present=data['C2']

# reshape into vectors and divide for training and testing
absent_train=absent[:900, :, :].reshape([900, -1])
present_train=present[:900, :, :].reshape([900, -1])
absent_test=absent[900:, :, :].reshape([100, -1])
present_test=present[900:, :, :].reshape([100, -1])

# normalize data
absent_train=normalize(np.transpose(absent_train))
present_train=normalize(np.transpose(present_train))
absent_test=normalize(np.transpose(absent_test))
present_test=normalize(np.transpose(present_test))

# random matrix for compressive sampling
rd=np.sign(np.random.randn(measurements, ps[0]*ps[1])/10.0)

# take compressed measurements with random matrix
absent_train=np.matmul(rd, absent_train)
present_train=np.matmul(rd, present_train)
absent_test=np.matmul(rd, absent_test)
present_test=np.matmul(rd, present_test)


with tf.Session() as sess:
  
  # learn fruit dictionary
  absent_dict, absent_alpha=LCA(absent_train, 200, 75, num_dict_features=k)
  
  

  # learn car dictionary
  present_dict, present_alpha=LCA(present_train, 200, 75, num_dict_features=k)
  present_dict=np.matmul(np.transpose(rd), present_dict)
  visualize_dict(present_dict, [20, 20], [36, 18])


  ##############################test new images#################################
  
  present_dict, present_alpha=LCA(present_test, 1, present_test.shape[1], D=absent_dict)
  present_dict2, present_alpha2=LCA(present_test, 1, present_test.shape[1], D=present_dict)

  







  
  

