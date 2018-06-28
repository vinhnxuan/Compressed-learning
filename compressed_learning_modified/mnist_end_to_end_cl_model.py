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
  
def setup_model(x, num_measurement, keep_prob):
    
    W_fc01 = weight_variable([784, num_measurement])

    h_fc01 = tf.nn.relu(tf.matmul(x, W_fc01))
    
    
    W_fc02 = weight_variable([num_measurement, 784])
    
    h_fc02 = tf.nn.relu(tf.matmul(h_fc01, W_fc02))
    
    
    x_image = tf.reshape(h_fc02, [-1, 28, 28, 1])
    
    # Convolutional layer 1
    W_conv1 = weight_variable([3, 3, 1,32])
    b_conv1 = bias_variable([32])
    
    #W_conv11 = weight_variable([3, 3, 1, 32])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)  + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # Convolutional layer 2
    W_conv2 = weight_variable([5, 5, 32, 48])
    b_conv2 = bias_variable([48])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_3x3(h_conv2)
    
    # Fully connected layer 1
    h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*48])
    
    W_fc1 = weight_variable([3*3*48, 150])
    b_fc1 = bias_variable([150])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    
    # Dropout
    #keep_prob  = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Fully connected layer 2 (Output layer)
    W_fc2 = weight_variable([150, 10])
    b_fc2 = bias_variable([10])
    
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')
    
    return x,y,W_fc01