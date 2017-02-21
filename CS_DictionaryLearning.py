import tensorflow as tf
import numpy as np
from scipy.misc import *
import os
import glob
from skimage.util import view_as_windows

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

def patchify(images, patch_sizes):
  return view_as_windows(images, patch_sizes)

def normalize(data):
  return (data-np.amin(data, axis=0))/((np.amax(data, axis=0)-np.amin(data, axis=0))+1e-6)

def LCA(y, num_dict_features, iters, batch_sz):
  D=np.random.rand(y.shape[0], num_dict_features)
  for i in range(iters):
    batch=y[:, np.uint8(np.floor(np.random.rand(batch_sz)*y.shape[1]))]
    batch=normalize(batch)
    D=tf.matmul(D, tf.diag(1/tf.sqrt(tf.reduce_sum(D**2, 0))))
    a=tf.matmul(tf.transpose(D), batch)
    a=tf.matmul(a, tf.diag(1/tf.sqrt(tf.reduce_sum(a**2, 0))))
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

ps=15
imsz=200
k=729

car_imgs=read_ims('/home/mpcr/Documents/MT/CSDL/photos/cars', imsz)
fruit_imgs=read_ims('/home/mpcr/Documents/MT/CSDL/photos/fruit', imsz)

car_patches=patchify(car_imgs, (1, ps, ps, 3))
fruit_patches=patchify(fruit_imgs, (1, ps, ps, 3))

car_patches=car_patches.reshape([car_patches.shape[0]*
				 car_patches.shape[1]*
				 car_patches.shape[2]*
				 car_patches.shape[3]*
				 car_patches.shape[4], -1])

fruit_patches=fruit_patches.reshape([fruit_patches.shape[0]*
				 fruit_patches.shape[1]*
				 fruit_patches.shape[2]*
				 fruit_patches.shape[3]*
				 fruit_patches.shape[4], -1])

car_patches=np.transpose(car_patches)
fruit_patches=np.transpose(fruit_patches)

rd=np.sign(np.random.rand(100, car_patches.shape[0]))

car_ms=np.matmul(rd, car_patches)
fruit_ms=np.matmul(rd, fruit_patches)

with tf.Session() as sess:
  car_dict, car_alpha=LCA(car_ms, k, 300, 32)
  fruit_dict, fruit_alpha=LCA(fruit_ms, k, 300, 32)
  car_dict=np.matmul(np.transpose(rd), car_dict)
  fruit_dict=np.matmul(np.transpose(rd), fruit_dict)
  print(car_dict.shape)
  print(car_dict)
  for i in range(324):
    imshow(imresize(np.reshape(car_dict[:, i], [15, 15, 3]), [50, 50]))

