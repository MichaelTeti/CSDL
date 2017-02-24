import tensorflow as tf
import numpy as np
from scipy.misc import *
import os
import glob
from skimage.util import view_as_windows
from scipy.io import loadmat
import functools


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


def doublewrap(function):
  """
  A decorator decorator, allowing to use the decorator to be used without 
  parentheses if not arguments are provided. All arguments must be optional.
  """
  @functools.wraps(function)
  def decorator(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
      return function(args[0])
    else:
      return lambda wrapee: function(wrapee, *args, **kwargs)
  return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
  """
  A decorator for functions that define TensorFlow operations. The wrapped
  function will only be executed once. Subsequent calls to it will directly
  return the result so that operations are added to the graph only once.
  The operations added by the function live within a tf.variable_scope(). If
  this decorator is used with arguments, they will be forwarded to the
  variable scope. The scope name defaults to the name of the wrapped
  function.
  """
  attribute = '_cache_' + function.__name__
  name = scope or function.__name__
  @property
  @functools.wraps(function)
  def decorator(self):
    if not hasattr(self, attribute):
      with tf.variable_scope(name, *args, **kwargs):
        setattr(self, attribute, function(self))
    return getattr(self, attribute)
  return decorator


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
    net=tf.matmul(self.X, tf.random_normal([400, 200], dtype=tf.float32))
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




ps=[36, 18]  # size of the images
measurements=300
k=400 # number of patches in dictionary

# read images from file and resize
data=loadmat('ped_images.mat')
absent=data['C1']
present=data['C2']

# reshape into vectors and divide for training and testing
absent_train=absent[:1500, :, :].reshape([1500, -1])
present_train=present[:1500, :, :].reshape([1500, -1])
train_data=np.concatenate((absent_train, present_train), axis=0)
absent_test=absent[1500:, :, :].reshape([3300, -1])
present_test=present[1500:, :, :].reshape([3300, -1])

# normalize data
train_data=normalize(np.transpose(train_data))
absent_test=normalize(np.transpose(absent_test))
present_test=normalize(np.transpose(present_test))

# random matrix for compressive sampling
rd=np.sign(np.random.randn(measurements, ps[0]*ps[1])/10.0)

# take compressed measurements with random matrix
train_data=np.matmul(rd, train_data)
absent_test=np.matmul(rd, absent_test)
present_test=np.matmul(rd, present_test)


with tf.Session() as sess:
  
  # learn dictionary
  print('Learning features of the data...')
  dict_, alpha_=LCA(train_data, 300, 75, num_dict_features=k)


##############################test new images#######################################

  batch_sz=200
  
  present_testd, present_testa=LCA(present_test, 1, present_test.shape[1], D=dict_)
  absent_testa, absent_testa=LCA(absent_test, 1, present_test.shape[1], D=dict_)
  elm_data=np.concatenate((present_testa.transpose(), absent_testa.transpose()), axis=0)
  elm_labels=np.concatenate((np.ones([3300, ]), np.zeros([3300, ])), axis=0)

  r=np.int32(np.floor(np.random.rand(6600)*6600))
  elm_testd=elm_data[r[6300:], :]
  elm_testl=elm_labels[r[6300:]]
  elm_data=elm_data[r[:6300], :]
  elm_labels=elm_labels[r[:6300]]

  x=tf.placeholder(dtype=tf.float32, shape=[None, k])
  y=tf.placeholder(dtype=tf.float32, shape=[None, ])

  elm=ELM(x, y)
  
  sess.run(tf.global_variables_initializer())
  a=0
  for i in range(100000):
    r=np.int32(np.floor(np.random.rand(batch_sz)*elm_data.shape[1]))
    sess.run(elm.optimize, {x: elm_data[r, :], y: elm_labels[r]})
    acc=sess.run(elm.accuracy, {x:elm_data[r, :], y: elm_labels[r]})
    print('Iteration: %d   acc: %f\r'%(i, acc))
    if i%100 == 0:
      print('val_acc: %f'%(sess.run(elm.accuracy, {x: elm_testd, y: elm_testl})))





