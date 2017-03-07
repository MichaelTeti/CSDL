import tensorflow as tf
import numpy as np
from scipy.misc import *
import os
from skimage.util import view_as_windows
from image_reader import *
import sys
import pickle


imsz=150
ps=8  # size of the images
measurements=80 # number of compressed measurements to take
k=250 # number of patches in dictionary
num_test_pics=60
num_classes=17

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

with tf.Session() as sess:

  try:
    rd=np.load('rand_matrix.npy')
    with open('flower_dicts.pickle', 'rb') as handle:
      d = pickle.load(handle)

  except IOError:
    d={}
  
    rd=np.random.randn(measurements, 3*ps**2)/10.0
    
    for i in range(num_classes):
  
      sys.stdout.write("Learning Dictionary %d / %d   \r" % (i+1, num_classes))
      sys.stdout.flush()

      patches=view_as_windows(data[i*80:i*80+20, :, :, :], (1, ps, ps, 3))

      patches=np.transpose(patches.reshape([patches.shape[0]*
			                    patches.shape[1]*
			       		    patches.shape[2]*
			       		    patches.shape[3]*
			       		    patches.shape[4], -1]))
 
      patches=np.matmul(rd, normalize(patches))

      #patches=np.matmul(rd.transpose(), patches)

      dict_, alpha_=LCA(patches, 400, 400, num_dict_features=k)

      d['dict{0}'.format(i)]=dict_
  
      #visualize_dict(dict_, d_shape=[12, 12], patch_shape=[ps, ps])

    with open('flower_dicts.pickle', 'wb') as handle:
      pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL) 
   
    np.save('rand_matrix.npy', rd)
  
    sys.exit(0)

################################ test new images #######################################

  val_acc=np.zeros([17*80])

  correct_label=np.zeros([17*80])

  for n in range(17):

    c1_test=data[n*80+20:n*80+20+num_test_pics, :, :, :]
   
    for i in range(num_test_pics):

      patches=view_as_windows(c1_test[i, :, :, :], (ps, ps, 3))
  
      patches=np.transpose(patches.reshape([patches.shape[0]*
	                                    patches.shape[1]*
	                                    patches.shape[2], -1]))


      patches=patches[:, np.int32(np.random.rand(7500)*patches.shape[1])]

      patches=np.matmul(rd, normalize(patches))  
  
      #patches=np.matmul(rd.transpose(), patches)

      best_dict=np.zeros([num_classes])

      for j in range(num_classes):
      
        testd=d['dict{0}'.format(j)] 
  
        c17td, c17ta=LCA(patches, 1, patches.shape[1], D=testd)

        best_dict[j]=np.sum(np.absolute(np.matmul(testd, c17ta)-patches))

      val_acc[n*num_test_pics+i]=np.argmin(best_dict)
 
      correct_label[n*num_test_pics+i]=np.argmax(labels[n*80+20:n*80+20+(i+1), :])

      sys.stdout.write('Test Image %d; Class: %d; Prediction: %d      \r' % (i+1, n, val_acc[i]) )
      sys.stdout.flush()

  correct_label=[float(x==y) for (x, y) in zip(val_acc, correct_label)]


  print('Correct: %f'%(np.mean(correct_label)))
