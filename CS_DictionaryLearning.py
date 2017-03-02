import tensorflow as tf
import numpy as np
from scipy.misc import *
import os
from skimage.util import view_as_windows
from image_reader import *
import h5py
import sys
import pickle


imsz=140
ps=8  # size of the images
measurements=110 # number of compressed measurements to take
k=400 # number of patches in dictionary
rd=np.sign(np.random.randn(measurements, 3*ps**2))
num_test_pics=60


def normalize(data):
  return (data-np.mean(data, axis=0))/(np.std(data, axis=0)+1e-6)


def LCA(y, iters, batch_sz, num_dict_features=None, D=None):
  ''' Dynamical systems neural network used for sparse approximation of an
      input vector.

      Args: 
           y: input signal or vector, or multiple column vectors.
           num_dict_features: number of dictionary patches to learn.
           iters: number of LCA iterations.
           batch_sz: number of samples to send to the network at each iteration.
           D: The dictionary to be used in the network. 
  '''
  assert(num_dict_features is None or D is None), 'provide D or num_dict_features, not both'
  if D is None:
    D=np.random.randn(y.shape[0], num_dict_features)
  for i in range(iters):
    batch=y[:, np.int32(np.floor(np.random.rand(batch_sz)*y.shape[1]))]
    D=tf.matmul(D, tf.diag(1/(tf.sqrt(tf.reduce_sum(D**2, 0))+1e-6)))
    a=tf.matmul(tf.transpose(D), batch)
    a=tf.matmul(a, tf.diag(1/(tf.sqrt(tf.reduce_sum(a**2, 0))+1e-6)))
    a=0.3*a**3
    D=D+tf.matmul((batch-tf.matmul(D, a)), tf.transpose(a))
  return sess.run(D), sess.run(a)


# read images from file and resize if not saved already
print('Loading Data...')


data, labels=read_ims('/home/mpcr/Documents/MT/CSDL/17flowers/jpg', imsz)

num_classes=labels.shape[1]

with tf.Session() as sess:

  try:
    with open('flower_dicts.pickle', 'rb') as handle:
      d = pickle.load(handle)
  except IOError:
    d={}

    for i in range(num_classes):
  
      sys.stdout.write("Learning Dictionary %d / %d   \r" % (i+1, num_classes) )
      sys.stdout.flush()

      patches=view_as_windows(data[i*80:i*80+20, :, :, :], (1, ps, ps, 3))

      patches=patches.reshape([patches.shape[0]*
			       patches.shape[1]*
			       patches.shape[2]*
			       patches.shape[3]*
			       patches.shape[4], -1])

      patches=normalize(patches.transpose())
 
      patches=np.matmul(rd, patches)

      dict_, alpha_=LCA(patches, 300, 300, num_dict_features=k)

      d['dict{0}'.format(i)]=dict_

      #dict_proj=np.matmul(rd.transpose(), dict_)
  
      #visualize_dict(dict_proj, d_shape=[20, 20, 3], patch_shape=[ps, ps])

    with open('flower_dicts.pickle', 'wb') as handle:
      pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL) 
  
    sys.exit(0)

################################ test new images #######################################
  

  num_classes=3

  c1_test=data[20:20+num_test_pics, :, :, :]
 
  ans=np.zeros([num_test_pics])
   
  for i in range(num_test_pics):

    sys.stdout.write('Test Image %d                    \r' % (i+1) )
    sys.stdout.flush()

    patches=view_as_windows(c1_test[i, :, :, :], (ps, ps, 3))
  
    patches=np.transpose(patches.reshape([patches.shape[0]*
	                                  patches.shape[1]*
	                                  patches.shape[2], -1]))


    patches=np.matmul(rd, patches[:, np.int32(np.random.rand(10000)*patches.shape[1])])

    patches=normalize(patches)

    ans1=np.zeros([num_classes])

    for j in range(num_classes):
      
      testd=d['dict{0}'.format(j)]
  
      c17td, c17ta=LCA(patches, 1, patches.shape[1], D=testd)

      ans1[j]=np.mean(np.absolute(np.matmul(testd, c17ta)-patches))
      print(ans1)

    ans[i]=np.argmin(ans1)

  correct=[float(x==y) for (x, y) in zip(ans, np.argmax(labels[20:80], axis=1))]


  print('Percent Correct: %f'%(np.mean(correct)))
