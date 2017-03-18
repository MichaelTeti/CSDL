import tensorflow as tf
import numpy as np
from scipy.misc import *
import os
from skimage.util import view_as_windows
from image_reader import *
import sys
import pickle


imsz=150
ps=16  # size of the images
measurements=100 # number of compressed measurements to take
k=400 # number of patches in first dictionary
num_classes=17
save=['csdl.h5', 'images', 'labels']
num_train=50
num_test_pics=80-num_train

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
try:
  f=h5py.File(save[0], 'r')
  data=f[save[1]]
  labels=f[save[2]]

  f=h5py.File('test_imgs_and_labels.h5', 'r')
  test_pics=f['test_imgs']
  correct_label=f['test_labels']


except IOError or KeyError:
  data, labels=read_ims('/home/mpcr/Documents/MT/CSDL/17flowers/jpg',
		        imsz, 
	                save=save)


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

      patches=view_as_windows(data[i*80:i*80+num_train, :, :, :], (1, ps, ps, 3))

      patches=np.transpose(patches.reshape([patches.shape[0]*
			                    patches.shape[1]*
			       		    patches.shape[2]*
			       		    patches.shape[3]*
			       		    patches.shape[4], -1]))
 
      patches=np.matmul(rd, normalize(patches))

      #patches=np.matmul(rd.transpose(), patches)

      dict_, alpha_=LCA(patches, 400, 100, num_dict_features=k)

      d['dict{0}'.format(i)]=dict_

      d['alpha{0}'.format(i)]=alpha_
  
      #visualize_dict(dict_, d_shape=[12, 12], patch_shape=[ps, ps])

    with open('flower_dicts.pickle', 'wb') as handle:
      pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL) 
   
    np.save('rand_matrix.npy', rd)

    testpics=np.zeros([num_test_pics*num_classes, imsz, imsz, 3])

    correct_label=np.zeros([num_classes*80])

    for j in range(num_classes):
      testdata=data[j*80+num_train:j*80+num_train+num_test_pics, :, :, :]
      testpics[j*num_test_pics:j*num_test_pics+num_test_pics, :, :, :]=testdata
      label=np.argmax(labels[j*80+num_train:j*80+num_train+num_test_pics, :], axis=1)
      correct_label[j*num_test_pics:j*num_test_pics+num_test_pics]=label

    f=h5py.File('test_imgs_and_labels.h5', 'a')
    f.create_dataset('test_imgs', data=testpics)
    f.create_dataset('test_labels', data=correct_label)
    f.close()

    sys.exit(0)

################################ test new images #######################################

  val_acc=np.zeros([num_classes*80])
   
  for i in range(test_pics.shape[0]):

    imshow(test_pics[i, :, :, :])

    patches=view_as_windows(test_pics[i, :, :, :], (ps, ps, 3))

    patches=patches[::4, ::4, :, :, :, :]
  
    patches=np.transpose(patches.reshape([patches.shape[0]*
	                                  patches.shape[1]*
	                                  patches.shape[2], -1]))


    patches=np.matmul(rd, normalize(patches))  
  
    #patches=np.matmul(rd.transpose(), patches)

    best_dict=np.zeros([num_classes])

    for j in range(num_classes):
      
      testd=d['dict{0}'.format(j)] 
 
      #testa=d['alpha{0}'.format(j)]
  
      c17td, c17ta=LCA(patches, 25, 100, D=testd)

      best_dict[j]=np.sum((np.matmul(testd, c17ta)-patches)**2)

    print(best_dict)

    val_acc[i]=np.argmax(best_dict)

    sys.stdout.write('Test Image %d; Class: %d; Prediction: %d      \r' % (i+1, np.floor(i/num_test_pics), val_acc[i]) )
    sys.stdout.flush()

  correct_label=[float(x==y) for (x, y) in zip(val_acc, correct_label)]


  print('Correct: %f'%(np.mean(correct_label)))
