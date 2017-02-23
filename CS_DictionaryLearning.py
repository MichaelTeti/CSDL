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

def visualize_dict(D, k, patch_shape, resize):
  d=np.zeros(np.sqrt(k)*patch_shape[0],
             np.sqrt(k)*patch_shape[1], 
             patch_shape[2])
  for row in range(np.sqrt(k)):
    for col in range(np.sqrt(k)):
      resized_patch=np.reshape(D[:, row*np.sqrt(k)+col], patch_shape)
      d[row*patch_shape[0]:row*patch_shape[0]+patch_shape[1],
        col*patch_shape[1]:col*patch_shape[1]+patch_shape[1],
        :] = resized_patch


ps=25  # size of one patch dimension
imsz=100  # size to reshape images to
k=625 # number of patches in dictionary

# read images from file and resize
car_imgs=read_ims('/home/mpcr/Documents/MT/CSDL/photos/cars', imsz)
fruit_imgs=read_ims('/home/mpcr/Documents/MT/CSDL/photos/fruit', imsz)

# average the color channels to get grayscale
car_imgs=np.mean(car_imgs, axis=3)
fruit_imgs=np.mean(fruit_imgs, axis=3)

# get 15x15 patches from each image
car_patches=view_as_windows(car_imgs, (1, ps, ps))
fruit_patches=view_as_windows(fruit_imgs, (1, ps, ps))

# reshape into vectors
car_patches=car_patches.reshape([car_patches.shape[0]*
				 car_patches.shape[1]*
				 car_patches.shape[2]*
				 car_patches.shape[3], -1])

fruit_patches=fruit_patches.reshape([fruit_patches.shape[0]*
				 fruit_patches.shape[1]*
				 fruit_patches.shape[2]*
				 fruit_patches.shape[3], -1])

car_patches=normalize(np.transpose(car_patches))
fruit_patches=normalize(np.transpose(fruit_patches))

# random matrix for compressive sampling
rd=np.sign(np.random.randn(100, ps**2)/10.0)

# take compressed measurements with random matrix
car_ms=np.matmul(rd, car_patches)
fruit_ms=np.matmul(rd, fruit_patches)


with tf.Session() as sess:
  
  # learn fruit dictionary
  fruit_dict, fruit_alpha=LCA(fruit_ms, 200, 65, num_dict_features=k)
  #fruit_dict=np.matmul(np.transpose(rd), fruit_dict)

  # learn car dictionary
  car_dict, car_alpha=LCA(car_ms, 200, 65, num_dict_features=k)
  #car_dict=np.matmul(np.transpose(rd), car_dict)

  #################test new photo of car#################################
  test_im=imread('apple.jpeg')
  test_im=imresize(test_im, [imsz, imsz])
  test_im=np.mean(test_im, axis=2)
  test_patches=view_as_windows(test_im, (ps, ps))
  test_patches=np.transpose(test_patches.reshape([test_patches.shape[0]*
                                                  test_patches.shape[1], -1]))
  
  test_patches=normalize(test_patches)
  test_meas=np.matmul(rd, test_patches)

  test_dict1, test_alpha1=LCA(test_meas, 1, test_meas.shape[1], D=fruit_dict)
  test_dict2, test_alpha2=LCA(test_meas, 1, test_meas.shape[1], D=car_dict)

  test_norm1=np.sum(test_alpha1)
  test_norm2=np.sum(test_alpha2)

  print('fruit norm: %f, car norm: %f'%(test_norm1, test_norm2))

  
  

