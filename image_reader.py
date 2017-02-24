import glob
import os
from scipy.misc import *
import numpy as np

def read_ims(directory, imsz):
  main_dir=os.getcwd()
  os.chdir(directory)
  num_ims=sum([len(files) for r, d, files in os.walk(directory)])
  imgs=np.zeros([num_ims, imsz, imsz, 3])
  labels=np.zeros([num_ims, len(os.listdir(os.getcwd()))])
  
  for f in os.listdir(os.getcwd()):
    print('Folder %s'%(f))
    os.chdir(f)
    r0=np.array(np.where(np.sum(labels, axis=1)==0))
    c0=np.array(np.where(np.sum(labels, axis=0)==0))
    labels[r0[0, 0]:r0[0, 0]+len(glob.glob1(os.getcwd(), '*')), c0[0, 0]]=1

    for filename in os.listdir(os.getcwd()):
      print(filename)
      im=imresize(imread(filename), [imsz, imsz])
      imgs[r0, :, :, :]=im
    os.chdir(directory)
  os.chdir(main_dir)
  return imgs, labels
