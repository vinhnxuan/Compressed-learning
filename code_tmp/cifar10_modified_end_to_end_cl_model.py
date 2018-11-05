#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:43:04 2018

@author: gw438
"""

import tensorflow as tf
import numpy as np

_IMAGE_SIZE = 32
_IMAGE_CHANNELS = 3
_NUM_CLASSES = 10
_RESHAPE_SIZE = 4*4*128
_NUM_CHANNELS = 3
_SAVE_PATH = "./tensorboard/cifar-10/"


def haarMatrix(n, normalized=False):
    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haarMatrix(n / 2, normalized)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    
    # calculate lower haar part 
    if normalized:

        h_n = np.kron(h, [1, 1])
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_n = np.kron(h, [1, 1])
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h






def variable_with_weight_decay( name, shape, stddev, wd):
        dtype = tf.float32
        var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

def variable_on_cpu(name, shape, initializer):
    
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var
  
def setup_model(x, num_measurement,keep_prob):
    
    H= haarMatrix(32*32,normalized=True).astype(np.float32)    
    x_2 = tf.reshape(x, [-1   , _IMAGE_CHANNELS , _IMAGE_SIZE, _IMAGE_SIZE], name='images')
    x_3 = tf.reshape(x_2, [-1  , _IMAGE_SIZE* _IMAGE_SIZE], name='images')

    var_list = []

    
    
    W_fc01 = variable_with_weight_decay('weights1', shape=[_IMAGE_SIZE* _IMAGE_SIZE, num_measurement], stddev=5e-2, wd=None)

    h_fc01 = (tf.matmul(x_3, W_fc01))
        
    #W_m= tf.transpose(tf.matmul(tf.transpose(W_fc01),tf.transpose(H)))
    W_m= W_fc01
    
    W_norm= tf.norm(W_m,ord='euclidean',axis=1,keepdims=True)
    norm_mat=tf.divide(W_m,W_norm)
    coherence=tf.matmul(norm_mat,tf.transpose(norm_mat))-tf.diag(tf.ones([_IMAGE_SIZE* _IMAGE_SIZE]))
    m_coherence=tf.reduce_max(tf.abs(coherence))
    
    x_1 = tf.reshape(h_fc01, [-1, _IMAGE_CHANNELS, num_measurement])
    
    
    W_fc02 = variable_with_weight_decay('weights2', shape=[_IMAGE_CHANNELS,num_measurement, 16*16*32], stddev=5e-2, wd=None)
    b_fc02 = variable_on_cpu('biases2', [16*16*32], tf.constant_initializer(0.0))
    
    h_fc02 = tf.nn.relu(tf.tensordot(x_1, W_fc02, axes=((1,2),(0,1)))+b_fc02)
    
    
    var_list.append(W_fc02)
    var_list.append(b_fc02)
    
    h_fc2_drop = tf.nn.dropout(h_fc02, keep_prob)
    
    x_image = tf.reshape(h_fc2_drop, [-1   , 16 , 16, 32], name='images')
    #x_image = tf.transpose(x_1, perm=[0, 2, 3, 1])
    
    
    kernel = variable_with_weight_decay('weights', shape=[5, 5, 32, 64], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
    biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name='conv1')
    
    var_list.append(kernel)
    var_list.append(biases)
        
    tf.summary.histogram('Convolution_layers/conv2', conv2)
    tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))
    
    

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

   
    kernel = variable_with_weight_decay('weights5', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = variable_on_cpu('biases5', [128], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation)
    var_list.append(kernel)
    var_list.append(biases)
        
    tf.summary.histogram('Convolution_layers/conv3', conv3)
    tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

    
    kernel = variable_with_weight_decay('weights6', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = variable_on_cpu('biases6', [128], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(pre_activation)#
    var_list.append(kernel)
    var_list.append(biases)
        
    tf.summary.histogram('Convolution_layers/conv4', conv4)
    tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))

    
    kernel = variable_with_weight_decay('weights7', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = variable_on_cpu('biases7', [128], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(pre_activation)
    var_list.append(kernel)
    var_list.append(biases)
        
    tf.summary.histogram('Convolution_layers/conv5', conv5)
    tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    
    reshape = tf.reshape(pool3, [-1, _RESHAPE_SIZE])
    dim = reshape.get_shape()[1].value
    weights = variable_with_weight_decay('weights8', shape=[dim, 384], stddev=0.04, wd=0.004)
    biases = variable_on_cpu('biases8', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases)
    var_list.append(weights)
    var_list.append(biases)
        
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    
    weights = variable_with_weight_decay('weights9', shape=[384, 192], stddev=0.04, wd=0.004)
    biases = variable_on_cpu('biases9', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases)
    var_list.append(weights)
    var_list.append(biases)
        
    tf.summary.histogram('Fully connected layers/fc2', local4)
    tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

    
    weights = variable_with_weight_decay('weights10', [192, _NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
    biases = variable_on_cpu('biases10', [_NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases)
    var_list.append(weights)
    var_list.append(biases)
        
    tf.summary.histogram('Fully connected layers/output', softmax_linear)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    y_pred_cls = tf.argmax(softmax_linear, axis=1)

    return x_1, softmax_linear, global_step, y_pred_cls, m_coherence, W_fc01, var_list
