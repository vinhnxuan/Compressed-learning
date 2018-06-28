#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:43:04 2018

@author: gw438
"""

import tensorflow as tf

_IMAGE_SIZE = 32
_IMAGE_CHANNELS = 3
_NUM_CLASSES = 10
_RESHAPE_SIZE = 4*4*128
_NUM_CHANNELS = 3
_SAVE_PATH = "./tensorboard/cifar-10/"


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
  
def setup_model(x, num_measurement):
    
    x_2 = tf.reshape(x, [-1   , _IMAGE_CHANNELS , _IMAGE_SIZE, _IMAGE_SIZE])
    x_3 = tf.reshape(x_2, [-1  , _IMAGE_SIZE* _IMAGE_SIZE])

    
    with tf.variable_scope('fully_connected01_128',reuse=tf.AUTO_REUSE):
        W_fc01 = variable_with_weight_decay('weights1', shape=[_IMAGE_SIZE* _IMAGE_SIZE, num_measurement], stddev=5e-2, wd=None)
        
        h_fc01 = tf.nn.relu(tf.matmul(x_3, W_fc01))
    
        W_norm= tf.norm(W_fc01,ord='euclidean',axis=1,keepdims=True)
        norm_mat=tf.divide(W_fc01,W_norm)
        coherence=tf.matmul(norm_mat,tf.transpose(norm_mat))-tf.diag(tf.ones([_IMAGE_SIZE* _IMAGE_SIZE]))
        #coherence_cost=tf.reduce_sum(tf.square(coherence))
    
        m_coherence=tf.reduce_max(tf.abs(coherence))
        
        
        
        W_fc02 = variable_with_weight_decay('weights2', shape=[num_measurement, _IMAGE_SIZE* _IMAGE_SIZE], stddev=5e-2, wd=None)
        b_fc02 = variable_on_cpu('biases2', [32*32], tf.constant_initializer(0.0))
        
        h_fc02 = tf.nn.relu(tf.matmul(h_fc01, W_fc02)+b_fc02)
    
    
    x_1 = tf.reshape(h_fc02, [-1   , _IMAGE_CHANNELS , _IMAGE_SIZE, _IMAGE_SIZE], name='images')
    x_image = tf.transpose(x_1, perm=[0, 2, 3, 1])
    #x_image = tf.transpose(x_1, perm=[0, 2, 3, 1])
    
    with tf.variable_scope('conv1_128',reuse=tf.AUTO_REUSE):
        kernel = variable_with_weight_decay('weights3', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases3', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation)
    tf.summary.histogram('Convolution_layers/conv1', conv1)
    tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))
    
    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    
    
    with tf.variable_scope('conv2_128',reuse=tf.AUTO_REUSE):
        kernel = variable_with_weight_decay('weights4', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases4', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation)
    tf.summary.histogram('Convolution_layers/conv2', conv2)
    tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3_128',reuse=tf.AUTO_REUSE):
        kernel = variable_with_weight_decay('weights5', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases5', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation)
    tf.summary.histogram('Convolution_layers/conv3', conv3)
    tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

    with tf.variable_scope('conv4_128',reuse=tf.AUTO_REUSE):
        kernel = variable_with_weight_decay('weights6', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases6', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation)
    tf.summary.histogram('Convolution_layers/conv4', conv4)
    tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))

    with tf.variable_scope('conv5_128',reuse=tf.AUTO_REUSE):
        kernel = variable_with_weight_decay('weights7', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases7', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation)
    tf.summary.histogram('Convolution_layers/conv5', conv5)
    tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('fully_connected1_128',reuse=tf.AUTO_REUSE):
        reshape = tf.reshape(pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights8', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases8', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases)
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    with tf.variable_scope('fully_connected2_128',reuse=tf.AUTO_REUSE):
        weights = variable_with_weight_decay('weights9', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases9', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases)
    tf.summary.histogram('Fully connected layers/fc2', local4)
    tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

    with tf.variable_scope('fully_connected3_128',reuse=tf.AUTO_REUSE):
        weights = variable_with_weight_decay('weights10', [192, _NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases10', [_NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases)
    tf.summary.histogram('Fully connected layers/output', softmax_linear)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    y_pred_cls = tf.argmax(softmax_linear, axis=1)

    return x, softmax_linear, global_step, y_pred_cls, m_coherence
