import glob
import os
from scipy.misc import *
import numpy as np
import cv2
import h5py

def read_ims(directory, imsz, grayscale=False, save=None):
  ''' Reads in images in subdirectories located in directory and 
      assigns a unique one-hot vector to each image in the respective
      folder.
      
      args:
           directory: the location of all the folders containing
                      each image class.
           imsz: resizes the width and height of each image to 
                 imsz
           grayscale: True if images are grayscale, False if color
                      images. Default is False.
           save: saves the images and labels as an h5 file. Arg is
                 a list with three strings containing the key for the 
                 data and the key for the labels. For example, 
                 ['images_labels.h5', 'images', 'labels']. 
                 Defaults to no saving. '''
 
  main_dir=os.getcwd()
  os.chdir(directory)
  if grayscale is True:
    num_channels=1
  else:
    num_channels=3
  num_ims=sum([len(files) for r, d, files in os.walk(directory)])
  imgs=np.zeros([num_ims, imsz, imsz, num_channels])
  labels=np.zeros([num_ims, len(os.listdir(os.getcwd()))])
  
  for f in os.listdir(os.getcwd()):
    print('Folder name: %s'%(f))
    os.chdir(f)
    r0=np.argmin(np.sum(labels, axis=1))
    c0=np.argmin(np.sum(labels, axis=0))
    labels[r0:r0+len(glob.glob1(os.getcwd(), '*')), c0]=1

    for filename in os.listdir(os.getcwd()):
      print(filename)
      im=imresize(imread(filename), [imsz, imsz])
      if im.ndim!=num_channels:
        print('Check %s file, wrong size'%(filename))
        sys.exit(0)
      imgs[r0, :, :, :]=im
    os.chdir(directory)
  os.chdir(main_dir)
  if save is not None:
    f=h5py.File(save[0], 'a')
    f.create_dataset(save[1], data=imgs)
    f.create_dataset(save[2], data=labels)
    f.close()
  return imgs, labels



def visualize_dict(D, d_shape, patch_shape):
  ''' Displays all sparse dictionary patches in one image.
      args:
           D: the sparse dictionary with size patch size x number of patches.
           d_shape: a list or tuple containing the desired number of patches per 
                    dimension of the dictionary. For example, a dictionary with
                    400 patches could be viewed at 20 patches x 20 patches.
           patch_shape: a list that specifies the width and height
                        to reshape each patch to. '''

  if np.size(d_shape)==2:
    vis_d=np.zeros([d_shape[0]*patch_shape[0], d_shape[1]*patch_shape[1], 1])
    resize_shp=[patch_shape[0], patch_shape[1], 1]
  else:
    vis_d=np.zeros([d_shape[0]*patch_shape[0], d_shape[1]*patch_shape[1], 3])
    resize_shp=[patch_shape[0], patch_shape[1], 3]

  for row in range(d_shape[0]):
    for col in range(d_shape[1]):
      resized_patch=np.reshape(D[:, row*d_shape[1]+col], resize_shp)
      vis_d[row*patch_shape[0]:row*patch_shape[0]+patch_shape[0], 
            col*patch_shape[1]:col*patch_shape[1]+patch_shape[1], :]=resized_patch
  if vis_d.shape[2]==3:
    imshow(vis_d)
  else:
    imshow(vis_d.reshape([vis_d.shape[0], vis_d.shape[1]]))
