# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:51:33 2018

@author: sgs4176
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Prepare Train data
train_x = np.linspace(-2, 2, 100)
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.33 + 10

#Define the model
X = tf.placeholder(shape=None, dtype=tf.float32)
Y = tf.placeholder(shape=None, dtype=tf.float32)

w = tf.Variable(0.0, name = 'weight')
b = tf.Variable(0.0, name = 'bais')

y_ = w * X + b
loss = tf.reduce_mean(tf.square(Y - y_))
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

#Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    epoch = 1
    for i in range(10):
        for (x, y) in zip(train_x, train_y):
            _, w_value, b_value, loss_value = sess.run([train_op, w, b,  loss], feed_dict={X: x, Y: y})
        print('Epoch:{}, w:{},b:{},loss:{}'.format(epoch, w_value, b_value, loss_value))
        epoch = epoch + 1
        
#draw the graph
plt.plot(train_x, train_y, '*')
plt.plot(train_x, train_x.dot(w_value)+b_value)
plt.show()