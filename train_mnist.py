from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
from scipy.misc import imshow, imresize


training_iters=8001
display_step=200
h_nodes=600
keep_prob=0.5
batch_size=200
m=250


def weights(shape, varname):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=varname)

def biases(shape, varname):
    return tf.Variable(tf.constant(0.0, shape=shape), name=varname)

def get_batch(data, labels, batch_size):
    rand=np.random.randint(0, data.shape[0], batch_size)
    return data[rand, :], labels[rand, :]


# Data loading and preprocessing
X=np.loadtxt('mnist_train.csv', delimiter=',')
Y=np.zeros([X.shape[0], 10])
for i in range(Y.shape[0]):
    Y[i, np.int32(X[i, 0])]=1
X=X[:, 1:].transpose()
n=X.shape[0]


testX=np.loadtxt('mnist_test.csv', delimiter=',')
testY=np.zeros([testX.shape[0], 10])
for i in range(testY.shape[0]):
    testY[i, np.int32(testX[i, 0])]=1
testX=testX[:, 1:].transpose()

X=(X-np.mean(X, 0))/(np.std(X, 0)+1e-6)
testX=(testX-np.mean(testX, 0))/(np.std(testX, 0)+1e-6)

A=np.random.randn(m, n)

X=np.matmul(A, X).transpose()
testX=np.matmul(A, testX).transpose()

# Create model
x=tf.placeholder(tf.float32, [None, m])
y=tf.placeholder(tf.float32, [None, 10])
drop=tf.placeholder(tf.float32)

w1=weights([m, h_nodes], 'w1')
net=tf.nn.dropout(tf.nn.tanh(tf.matmul(x, w1)+biases([h_nodes], 'b1')), drop)

w2=weights([h_nodes, h_nodes], 'w2')
net=tf.nn.dropout(tf.nn.tanh(tf.matmul(net, w2)+biases([h_nodes], 'b2')), drop)

w3=weights([h_nodes, h_nodes], 'w3')
net=tf.nn.dropout(tf.nn.tanh(tf.matmul(net, w3)+biases([h_nodes], 'b3')), drop)

w4=weights([h_nodes, 10], 'w4')
net=tf.matmul(net, w4)+biases([1], 'b4')

#cost = -tf.reduce_sum(y*tf.log(tf.clip_by_value(net,1e-10,1.0)))
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net))
cost=.005*tf.nn.l2_loss(w1)+.005*tf.nn.l2_loss(w2)+.005*tf.nn.l2_loss(w3)+.005*tf.nn.l2_loss(w4)+cost
train=tf.train.AdamOptimizer(1e-3).minimize(cost)
correct=tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y, 1), tf.argmax(net, 1))))

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(training_iters):
        d, l=get_batch(X, Y, batch_size)
        Cost, Correct=sess.run([cost, correct], feed_dict={x:d, y:l, drop:1.0})
        print('Iteration: %d, Cost: %f, Training Acc.: %f'%(i, Cost, Correct))
        sess.run(train, feed_dict={x:d, y:l, drop:keep_prob})

        if i%display_step==0:
            correct_val=sess.run(correct, feed_dict={x:testX, y:testY, drop:1.0})
            print('Validation Acc.: %f'%(correct_val))

    #save_variables=saver.save(sess, 'mnist_realGAN.ckpt')
