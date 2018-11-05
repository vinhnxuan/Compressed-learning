#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:43:04 2018

@author: gw438
"""

import tensorflow as tf


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')
  
def setup_model(x, num_measurement, keep_prob, keep_prob2):
    
    W_fc01 = weight_variable([784, num_measurement])

    h_fc01 = (tf.matmul(x, W_fc01))
    
    var_list=[]

    var_list.append(W_fc01)
    
    
    W_fc02 = weight_variable([num_measurement, 9*9*32])
    b_fc02 = bias_variable([9*9*32])
    
    h_fc02 = tf.nn.relu(tf.matmul(h_fc01, W_fc02)+b_fc02)
    
    var_list.append(W_fc02)
    var_list.append(b_fc02)
    
    
    h_fc2_drop = tf.nn.dropout(h_fc02, keep_prob2)
    
    
    x_image = tf.reshape(h_fc2_drop, [-1, 9, 9, 32])
    
    #x_image = gaussian_noise_layer(x_image, .2)
    #x_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x_image)
    
    
    # Convolutional layer 2
    W_conv2 = weight_variable([4, 4, 32, 48])
    b_conv2 = bias_variable([48])
    
    var_list.append(W_conv2)
    var_list.append(b_conv2)
    
    h_conv2 = tf.nn.relu(conv2d(x_image, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # Fully connected layer 1
    h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*48])
    
    W_fc1 = weight_variable([3*3*48, 150])
    b_fc1 = bias_variable([150])
    
    var_list.append(W_fc1)
    var_list.append(b_fc1)
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    
    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Fully connected layer 2 (Output layer)
    W_fc2 = weight_variable([150, 10])
    b_fc2 = bias_variable([10])
    
    var_list.append(W_fc2)
    var_list.append(b_fc2)
    
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')
    
    return x,y,W_fc01, var_list