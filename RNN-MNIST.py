# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:18:43 2018

@author: bharat
"""

import tensorflow as tf
from tensorflow.contrib import rnn

#import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)


#unrolled through 28 time steps
time_steps=28
#hidden LSTM units
num_units=128

n_input=28

learning_rate=0.001

n_classes=10
batch_size=128

#Weights
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))


#placeholder for input and output
x=tf.placeholder("float",[None,time_steps,n_input])

y=tf.placeholder("float",[None,n_classes])

"""[batch_size,time_steps,n_input],   we need to
 convert it into a list of tensors of shape    [batch_size,n_inputs]     of length time_steps 
 so that it can be then fed to static_rnn"""
 
input =tf.unstack(x ,time_steps,1)


lstm_layer = rnn.BasicLSTMCell(num_units , forget_bias =1.0)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")


#their are n_units outputs but we want n_classes outputs only(Last 10)
prediction=tf.matmul(outputs[-1],out_weights)+out_bias

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Running the session
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    while iter<800:
        #next_batch(batch_size)===784
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
        #We have to reshape it to [batch_size,time_steps , n_input]
        batch_x=batch_x.reshape((batch_size,time_steps,n_input))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1