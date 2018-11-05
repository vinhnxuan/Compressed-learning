#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:43:04 2018

@author: gw438
"""

import tensorflow as tf
from scipy.linalg import dft
import numpy as np


image_s =256


def layer_vectors(layers,n_nodes):
    
    layer_vecs= {}
    for i in range(n_nodes):
        layer_vecs[i]=tf.reshape(layers[i], [-1, image_s*image_s])
    return layer_vecs

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding='VALID')


def conv2d_same(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
def max_pool_2x1(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1], padding='SAME')
  
def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')
  
def max_pool_3x1(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 1, 1],
                        strides=[1, 3, 1, 1], padding='SAME')
  

  
def setup_model(x, num_measurement, keep_prob,keep_prob_2):
    
#    W_fc01 = weight_variable([784, num_measurement])
#
#    h_fc01 = tf.nn.relu(tf.matmul(x, W_fc01))
#    
#    
#    W_fc02 = weight_variable([num_measurement, 784])
#    
#    h_fc02 = tf.nn.relu(tf.matmul(h_fc01, W_fc02))
    
    n= x.get_shape()[2].value
    num_measurement=6
    
    #f = np.arange(num_measurement).reshape([1,num_measurement])
    
    n_vector=np.arange(n)
    
    n_array=n_vector.reshape([n,1])
    
    np.random.shuffle(n_vector)
    
    #f= n_vector[0:num_measurement].reshape([1,num_measurement])
    
    f = [1 ,5 ,11 ,24 ,25 ,27]
    
    f = np.asarray(f)
    
    f = f.reshape([1,num_measurement])
    

    fc01 = tf.Variable(f,dtype=tf.float32)
    nc01 = tf.constant(n_array,dtype=tf.float32)
    
    kk= tf.matmul(nc01,tf.ceil(fc01))
    
    kk = tf.cast(kk, tf.complex64)
    
    dft_matrix= tf.exp(-2j*np.pi*kk/n)/np.sqrt(num_measurement)
    
#    n_vector=np.ones(n)
#    
#    fc01 = tf.Variable(n_vector,dtype=tf.float32)
#    
#    fc01_drop= tf.nn.dropout(fc01, keep_prob_2)
#    
#    fc02 = tf.cast(fc01_drop, tf.complex64)
#    
#    dftn = tf.cast(dft(n),tf.complex64)
#    
#    
#    dft_matrix= tf.matmul(tf.diag(fc02),dftn)/np.sqrt(n)
    #dft_matrix= tf.constant(dft_matrix, dtype=tf.complex64)
#    
    x = tf.cast(x, dtype=tf.complex64)
    
    dft_x=tf.tensordot(x,dft_matrix,axes=[[2],[0]])
    
    dft_x = tf.reshape(dft_x, [-1, num_measurement*n])
    #output_dim= 14*
    
    W_fc01 = weight_variable([num_measurement*n, 9*9*32])
    b_fc01 = weight_variable([9*9*32])
    
    W_fc02 = weight_variable([num_measurement*n,9*9*32])
    b_fc02 = weight_variable([9*9*32])
    
    W = tf.complex(W_fc01,W_fc02)
    b = tf.complex(b_fc01,b_fc02)

    
    h_fc01 = tf.nn.relu(tf.real(tf.matmul(dft_x, W )+b))
    
    #h_fc01_drop = tf.nn.dropout(h_fc01, keep_prob_2)
    
    x_image = tf.reshape(h_fc01, [-1, 9, 9, 32])
    
    
    # Convolutional layer 2
    W_conv2 = weight_variable([4, 4, 32, 48])
    b_conv2 = bias_variable([48])
    
    h_conv2 = tf.nn.relu(conv2d(x_image, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # Convolutional layer 1
#    W_conv1 = weight_variable([3, 3, 1,32])
#    b_conv1 = bias_variable([32])
#    
#    #W_conv11 = weight_variable([3, 3, 1, 32])
#    
#    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)  + b_conv1)
#    h_pool1 = max_pool_2x2(h_conv1)
#    
#    # Convolutional layer 2
#    W_conv2 = weight_variable([3, 3, 32, 32])
#    b_conv2 = bias_variable([32])
#    
#    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#    h_pool2 = max_pool_2x2(h_conv2)
#    
#    
#    W_conv3 = weight_variable([3, 3, 32, 64])
#    b_conv3 = bias_variable([64])
#    
#    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#    h_pool3 = max_pool_2x2(h_conv3)
#    
#    W_conv4 = weight_variable([3, 3, 64, 128])
#    b_conv4 = bias_variable([128])
#    
#    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
#    h_pool4 = max_pool_2x2(h_conv4)
    
    
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
    
    return x,y,W_conv2,fc01


def setup_model2d(x, num_measurement, keep_prob,keep_prob_2):
    
#    W_fc01 = weight_variable([784, num_measurement])
#
#    h_fc01 = tf.nn.relu(tf.matmul(x, W_fc01))
#    
#    
#    W_fc02 = weight_variable([num_measurement, 784])
#    
#    h_fc02 = tf.nn.relu(tf.matmul(h_fc01, W_fc02))
    
    n= x.get_shape()[2].value
    num_measurement=10
    

    
#    n_vector=np.ones(n)
#    
#    fc01 = tf.Variable(n_vector,dtype=tf.float32)
#    
#    fc01_drop= tf.nn.dropout(fc01, keep_prob_2)
#    
#    fc02 = tf.cast(fc01_drop, tf.complex64)
#     
    dft_matrix= tf.constant(dft(n)/np.sqrt(n), dtype=tf.complex64)
    
    dft_mat = tf.contrib.kfac.utils.kronecker_product(tf.transpose(dft_matrix),dft_matrix)
#    
    x = tf.reshape(x,[-1,n*n])
    
    
    x = tf.cast(x, dtype=tf.complex64)
    
    dft_x=tf.matmul(x,dft_mat)
    
    W_fc03 = weight_variable([n*n, num_measurement])
    
    W_fc03_complex = tf.cast(W_fc03, dtype=tf.complex64)
    
    dft_y=tf.matmul(dft_x,W_fc03_complex)
    
    
    
    
    W_fc01 = weight_variable([num_measurement,n*n])
    b_fc01 = weight_variable([n*n])
    
    W_fc02 = weight_variable([num_measurement,n*n])
    b_fc02 = weight_variable([n*n])
    
    W = tf.complex(W_fc01,W_fc02)
    b = tf.complex(b_fc01,b_fc02)

    
    h_fc01 = tf.nn.relu(tf.real(tf.matmul(dft_y, W )))
    
    #h_fc01_drop = tf.nn.dropout(h_fc01, keep_prob_2)
    
    x_image = tf.reshape(h_fc01, [-1, n, n, 1])
    
    W_conv1 = weight_variable([4, 4, 1,32])
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)  + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # Convolutional layer 2
    W_conv2 = weight_variable([6, 6, 32, 48])
    b_conv2 = bias_variable([48])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_3x3(h_conv2)
    

    
    # Convolutional layer 1
#    W_conv1 = weight_variable([3, 3, 1,32])
#    b_conv1 = bias_variable([32])
#    
#    #W_conv11 = weight_variable([3, 3, 1, 32])
#    
#    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)  + b_conv1)
#    h_pool1 = max_pool_2x2(h_conv1)
#    
#    # Convolutional layer 2
#    W_conv2 = weight_variable([3, 3, 32, 32])
#    b_conv2 = bias_variable([32])
#    
#    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#    h_pool2 = max_pool_2x2(h_conv2)
#    
#    
#    W_conv3 = weight_variable([3, 3, 32, 64])
#    b_conv3 = bias_variable([64])
#    
#    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#    h_pool3 = max_pool_2x2(h_conv3)
#    
#    W_conv4 = weight_variable([3, 3, 64, 128])
#    b_conv4 = bias_variable([128])
#    
#    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
#    h_pool4 = max_pool_2x2(h_conv4)
    
    
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
    
    return x,y,W_conv2,W_fc03


def setup_model_cs2(x, num_measurement, keep_prob,keep_prob_2):
    
#    W_fc01 = weight_variable([784, num_measurement])
#
#    h_fc01 = tf.nn.relu(tf.matmul(x, W_fc01))
#    
#    
#    W_fc02 = weight_variable([num_measurement, 784])
#    
#    h_fc02 = tf.nn.relu(tf.matmul(h_fc01, W_fc02))
    
    n= x.get_shape()[2].value
    num_measurement=n
    
    #f = np.arange(num_measurement).reshape([1,num_measurement])
    
    n_vector=np.arange(n)
    
    n_array=n_vector.reshape([n,1])
    
    np.random.shuffle(n_vector)
    
    f= n_vector[0:num_measurement].reshape([1,num_measurement])
    
    #f = [1 ,5 ,11 ,24 ,25 ,27]
    
    #f = np.asarray(f)
    
    #f = f.reshape([1,num_measurement])
    

    fc01 = tf.Variable(f,dtype=tf.float32)
    nc01 = tf.constant(n_array,dtype=tf.float32)
    
    kk= tf.matmul(nc01,tf.ceil(fc01))
    
    kk = tf.cast(kk, tf.complex64)
    
    dft_matrix= tf.exp(-2j*np.pi*kk/n)/np.sqrt(num_measurement)
    
#    n_vector=np.ones(n)
#    
#    fc01 = tf.Variable(n_vector,dtype=tf.float32)
#    
#    fc01_drop= tf.nn.dropout(fc01, keep_prob_2)
#    
#    fc02 = tf.cast(fc01_drop, tf.complex64)
#    
#    dftn = tf.cast(dft(n),tf.complex64)
#    
#    
#    dft_matrix= tf.matmul(tf.diag(fc02),dftn)/np.sqrt(n)
    #dft_matrix= tf.constant(dft_matrix, dtype=tf.complex64)
#    
#    x = tf.cast(x, dtype=tf.complex64)
#    
#    dft_x=tf.tensordot(x,dft_matrix,axes=[[2],[0]])
#    
#    
#    #output_dim= 14*
#    
#    W_fc01 = weight_variable([num_measurement, n])
#    
#    W_fc02 = weight_variable([num_measurement,n])
#    
#    W = tf.complex(W_fc01,W_fc02)
#
#    
#    h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_x, tf.transpose(dft_matrix,conjugate=True) ,axes=[[2],[0]])))
#    
    x_image = tf.reshape(x, [-1, n, n, 1])
    
    W_conv1 = weight_variable([4, 4, 1, 32])
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)  + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # Convolutional layer 2
    W_conv2 = weight_variable([6, 6, 32, 48])
    b_conv2 = bias_variable([48])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_3x3(h_conv2)
    
    # Convolutional layer 1
#    W_conv1 = weight_variable([3, 3, 1,32])
#    b_conv1 = bias_variable([32])
#    
#    #W_conv11 = weight_variable([3, 3, 1, 32])
#    
#    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)  + b_conv1)
#    h_pool1 = max_pool_2x2(h_conv1)
#    
#    # Convolutional layer 2
#    W_conv2 = weight_variable([3, 3, 32, 32])
#    b_conv2 = bias_variable([32])
#    
#    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#    h_pool2 = max_pool_2x2(h_conv2)
#    
#    
#    W_conv3 = weight_variable([3, 3, 32, 64])
#    b_conv3 = bias_variable([64])
#    
#    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#    h_pool3 = max_pool_2x2(h_conv3)
#    
#    W_conv4 = weight_variable([3, 3, 64, 128])
#    b_conv4 = bias_variable([128])
#    
#    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
#    h_pool4 = max_pool_2x2(h_conv4)
    
    
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
    
    return x,y,W_conv1,fc01

def setup_model_original(x, keep_prob,keep_prob_2):
    
    
    n= x.get_shape()[2].value
    
    x_image = tf.reshape(x, [-1, n, n, 1])
    
    W_conv1 = weight_variable([4, 4, 1, 32])
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)  + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    
    # Convolutional layer 2
    W_conv2 = weight_variable([6, 6, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_3x3(h_conv2)
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*64])
    
    W_fc1 = weight_variable([3*3*64, 150])
    b_fc1 = bias_variable([150])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    
    # Dropout
    #keep_prob  = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Fully connected layer 2 (Output layer)
    W_fc2 = weight_variable([150, 10])
    b_fc2 = bias_variable([10])
    
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')
    
    return x,y,W_conv1,b_fc1


def setup_model_autofocus(x, keep_prob,keep_prob_2,phase_errors,nums):
    
    
    n= x.get_shape()[2].value
    
    x = x/255.
    
    dft_matrix= tf.constant(dft(n)/np.sqrt(n), dtype=tf.complex64)
    
    x = tf.cast(x, dtype=tf.complex64)
    
    dft_x=tf.tensordot(x,dft_matrix,axes=[[2],[0]])
    
    aa=tf.cast(tf.multiply(np.pi,phase_errors), dtype=tf.complex64)
    
    phase_error_amp= tf.exp(-2j*aa)
    
    multiply = tf.constant([1,n])
    
    phase_error_matrix = tf.reshape(tf.tile(phase_error_amp, multiply), [-1, n, n])
     
    phase_error_matrix = tf.cast(phase_error_matrix, dtype=tf.complex64)
    
    dft_y=tf.multiply(dft_x,tf.transpose(phase_error_matrix,perm=[0,2,1]))
    
    
    W_fc01 = weight_variable([n, n])
    
    W_fc02 = weight_variable([n, n])
    
    W = tf.complex(W_fc01,W_fc02)

    
    h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y, W ,axes=[[2],[0]])))
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y, tf.transpose(dft_matrix,conjugate=True) ,axes=[[2],[0]])))
    
    x_image = tf.reshape(h_fc01, [-1, n, n, 1])
    
    residual_layers= deep_residual_conv_add(x_image,nums,keep_prob)

    residual_vectors= layer_vectors(residual_layers,nums)

    
    return x,residual_layers,residual_vectors,h_fc01


def gen_measurement_cs(x,phase_errors, speck_errors,batch_cs_matrix):
    n= x.get_shape()[2].value
    
    x=x
    
    dft_m = (dft(n)/np.sqrt(n))
    
    dft_matrix= tf.constant(dft_m, dtype=tf.complex64)
    
    x_speckled=tf.multiply(x, speck_errors)
    
    x_speckled = tf.cast(x_speckled, dtype=tf.complex64)
    
    dft_x=tf.tensordot(x_speckled,dft_matrix,axes=[[2],[0]])
    
    aa=tf.cast(tf.multiply(np.pi,phase_errors), dtype=tf.complex64)
    
    phase_error_amp= tf.exp(-2j*aa)
    
    multiply = tf.constant([1,n])
    
    phase_error_matrix = tf.reshape(tf.tile(phase_error_amp, multiply), [-1, n, n])
     
    phase_error_matrix = tf.cast(phase_error_matrix, dtype=tf.complex64)
    
    dft_y=tf.multiply(dft_x,tf.transpose(phase_error_matrix,perm=[0,2,1]))

    batch_cs_matrix = tf.cast(batch_cs_matrix, dtype=tf.complex64)
    dft_y_cs=tf.multiply(dft_y,batch_cs_matrix)
    
    return dft_y_cs,dft_matrix


def setup_model_autofocus_cs(x, keep_prob,keep_prob_2,phase_errors, speck_errors,nums, batch_cs_matrix):
    
    
    n= x.get_shape()[2].value
    
    x=x
    
    dft_m = (dft(n)/np.sqrt(n))
    
    dft_matrix= tf.constant(dft_m, dtype=tf.complex64)
    
    x_speckled=tf.multiply(x, speck_errors)
    
    x_speckled = tf.cast(x_speckled, dtype=tf.complex64)
    
    dft_x=tf.tensordot(x_speckled,dft_matrix,axes=[[2],[0]])
    
    aa=tf.cast(tf.multiply(np.pi,phase_errors), dtype=tf.complex64)
    
    phase_error_amp= tf.exp(-2j*aa)
    
    multiply = tf.constant([1,n])
    
    phase_error_matrix = tf.reshape(tf.tile(phase_error_amp, multiply), [-1, n, n])
     
    phase_error_matrix = tf.cast(phase_error_matrix, dtype=tf.complex64)
    
    dft_y=tf.multiply(dft_x,tf.transpose(phase_error_matrix,perm=[0,2,1]))

    batch_cs_matrix = tf.cast(batch_cs_matrix, dtype=tf.complex64)
    dft_y_cs=tf.multiply(dft_y,batch_cs_matrix)
    
    #batch_cs_matrix = weight_variable([n*n])
    
    #batch_cs_matrix_drop = tf.nn.dropout(batch_cs_matrix, keep_prob_2)
    
#    batch_cs_matrix_drop = tf.cast(batch_cs_matrix_drop, dtype=tf.complex64)
#    
#    dft_y= tf.reshape(dft_y,[-1,n*n])
#    
#    dft_y_cs=tf.einsum('ai,i->ai',dft_y,batch_cs_matrix_drop)
#    
#    dft_y_cs= tf.reshape(dft_y_cs,[-1,n,n])
    
    #dft_y_cs=tf.multiply(dft_y,tf.transpose(batch_cs_matrix,perm=[0,2,1]))
    
    
    
#    dft_y_cs_transpose= tf.transpose(dft_y_cs,perm=[1,0,2])
#    
#    W_fc01 = {}
#    W_fc02 = {}
#    
#    W={}
#    
#    for i in range(n):
#        
#        
#
#        W_fc01[i] = weight_variable([n, n])
#    
#        W_fc02[i] = weight_variable([n, n])
#    
#        W[i] = tf.complex(W_fc01[i],W_fc02[i])
#        
#        row_dft_y_cs= tf.gather_nd(dft_y_cs_transpose,[[i]])
#        
#        row_dft_y_cs= tf.reshape(row_dft_y_cs,[-1,1,n])
#        
#        if i ==0:
#            h_fc01 = tf.nn.relu(tf.real(tf.tensordot(row_dft_y_cs, W[i] ,axes=[[2],[0]])))
#        else:
#            tmp = tf.nn.relu(tf.real(tf.tensordot(row_dft_y_cs, W[i] ,axes=[[2],[0]])))
#            h_fc01 = tf.concat([h_fc01,tmp],1)
    
    
    W_fc01 = weight_variable([n, n])
    
    W_fc02 = weight_variable([n, n])
    
    W = tf.complex(W_fc01,W_fc02)

    
    h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, W ,axes=[[2],[0]])))
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, W ,axes=[[2],[0]])))
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, tf.transpose(dft_matrix,conjugate=True) ,axes=[[2],[0]])))
    
    x_image = tf.reshape(h_fc01, [-1, n, n, 1])
    
    #residual_layers= deep_residual_conv_add(x_image,nums,keep_prob)
    
    residual_layers= deep_residual_ista_mix_speckling(x_image,nums,keep_prob)

    residual_vectors= layer_vectors(residual_layers,nums)

    
    return x,residual_layers,residual_vectors,dft_y


def setup_model_autofocus_cs_model_2(x, keep_prob,keep_prob_2,phase_errors, speck_errors,nums, batch_cs_matrix,f1,f2):
    
    
    n= x.get_shape()[2].value
    
    x=x
    
    dft_m = (dft(n)/np.sqrt(n))
    
    
    dft_matrix= tf.constant(dft_m, dtype=tf.complex64)
    
    dft_matrix_h =tf.transpose(dft_matrix,conjugate=True)
    
    x_speckled=tf.multiply(x, speck_errors)
    
    x_speckled = tf.cast(x_speckled, dtype=tf.complex64)
    
    dft_x=tf.tensordot(x_speckled,dft_matrix,axes=[[2],[0]])
    
    dft_x_2=tf.transpose(tf.tensordot(tf.transpose(dft_x,perm=[0,2,1]),tf.transpose(dft_matrix_h),axes=[[2],[0]]),perm=[0,2,1])
    
    
    aa=tf.cast(tf.multiply(np.pi,phase_errors), dtype=tf.complex64)
    
    phase_error_amp= tf.exp(-2j*aa)
    
    multiply = tf.constant([1,n])
    
    phase_error_matrix = tf.reshape(tf.tile(phase_error_amp, multiply), [-1, n, n])
     
    phase_error_matrix = tf.cast(phase_error_matrix, dtype=tf.complex64)
    
    dft_y=tf.multiply(dft_x_2,tf.transpose(phase_error_matrix,perm=[0,2,1]))
    
    
    
    

    batch_cs_matrix = tf.cast(batch_cs_matrix, dtype=tf.complex64)
    dft_y_cs=tf.multiply(dft_y,batch_cs_matrix)
    
    #batch_cs_matrix = weight_variable([n*n])
    
    #batch_cs_matrix_drop = tf.nn.dropout(batch_cs_matrix, keep_prob_2)
    
#    batch_cs_matrix_drop = tf.cast(batch_cs_matrix_drop, dtype=tf.complex64)
#    
#    dft_y= tf.reshape(dft_y,[-1,n*n])
#    
#    dft_y_cs=tf.einsum('ai,i->ai',dft_y,batch_cs_matrix_drop)
#    
#    dft_y_cs= tf.reshape(dft_y_cs,[-1,n,n])
    
    #dft_y_cs=tf.multiply(dft_y,tf.transpose(batch_cs_matrix,perm=[0,2,1]))
    
    
    
#    dft_y_cs_transpose= tf.transpose(dft_y_cs,perm=[1,0,2])
#    
#    W_fc01 = {}
#    W_fc02 = {}
#    
#    W={}
#    
#    for i in range(n):
#        
#        
#
#        W_fc01[i] = weight_variable([n, n])
#    
#        W_fc02[i] = weight_variable([n, n])
#    
#        W[i] = tf.complex(W_fc01[i],W_fc02[i])
#        
#        row_dft_y_cs= tf.gather_nd(dft_y_cs_transpose,[[i]])
#        
#        row_dft_y_cs= tf.reshape(row_dft_y_cs,[-1,1,n])
#        
#        if i ==0:
#            h_fc01 = tf.nn.relu(tf.real(tf.tensordot(row_dft_y_cs, W[i] ,axes=[[2],[0]])))
#        else:
#            tmp = tf.nn.relu(tf.real(tf.tensordot(row_dft_y_cs, W[i] ,axes=[[2],[0]])))
#            h_fc01 = tf.concat([h_fc01,tmp],1)
    
    
    W_fc01 = weight_variable([n, n])
    
    W_fc02 = weight_variable([n, n])

    W = tf.complex(W_fc01,W_fc02)
    
    a1= np.zeros([n,n])
    
    a1[f2,:]=1
    
    a1 = tf.constant(a1,dtype=tf.complex64)
    
    a1_mul= tf.multiply(W,a1)
    #W = dft_matrix_h
    
    W_fc03 = weight_variable([n, n])
    
    W_fc04 = weight_variable([n, n])
    
    W2 = tf.complex(W_fc03,W_fc04)
    
    a2= np.zeros([n,n])
    
    a2[:,f1]=1
    
    a2 = tf.constant(a2,dtype=tf.complex64)
    
    a2_mul= tf.multiply(W2,a2)
    
    #W2 = tf.transpose(W, conjugate=True)
    
    y1= tf.tensordot(dft_y_cs, W ,axes=[[2],[0]])
    
    y2= tf.transpose(tf.tensordot(tf.transpose(y1,perm=[0,2,1]),tf.transpose(W2),axes=[[2],[0]]),perm=[0,2,1])

    
    h_fc01 = tf.nn.relu(tf.abs(y2))
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, W ,axes=[[2],[0]])))
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, tf.transpose(dft_matrix,conjugate=True) ,axes=[[2],[0]])))
    
    x_image = tf.reshape(h_fc01, [-1, n, n, 1])
    
    #residual_layers= deep_residual_conv_add(x_image,nums,keep_prob)
    
    residual_layers= deep_residual_ista_mix_speckling(x_image,nums,keep_prob)

    residual_vectors= layer_vectors(residual_layers,nums)

    
    return x,residual_layers,residual_vectors,y2



def setup_model_autofocus_on_complex(x, keep_prob,keep_prob_2,phase_errors,nums):
    
    
    n= x.get_shape()[2].value
    
    dft_matrix= tf.constant(dft(n)/np.sqrt(n), dtype=tf.complex64)
    
    x = tf.cast(x, dtype=tf.complex64)
    
    dft_x=tf.tensordot(x,dft_matrix,axes=[[2],[0]])
    
    aa=tf.cast(tf.multiply(np.pi,phase_errors), dtype=tf.complex64)
    
    phase_error_amp= tf.exp(-2j*aa)
    
    multiply = tf.constant([1,n])
    
    phase_error_matrix = tf.reshape(tf.tile(phase_error_amp, multiply), [-1, n, n])
     
    phase_error_matrix = tf.cast(phase_error_matrix, dtype=tf.complex64)
    
    dft_y=tf.multiply(dft_x,tf.transpose(phase_error_matrix,perm=[0,2,1]))

    
    h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y, tf.transpose(dft_matrix,conjugate=True) ,axes=[[2],[0]])))
#    
#    
#    
    x_image = tf.reshape(dft_y, [-1, n, n])
    
    residual_layers= deep_residual_ista_mix_complex(x_image,nums,keep_prob)

    residual_vectors= layer_vectors(residual_layers,nums)

    
    return x,residual_layers,residual_vectors,h_fc01


def setup_model_speckling(x, keep_prob,keep_prob_2,phase_errors,nums):
    
    
    
    n= x.get_shape()[2].value
    
    #
    x_processed = (x/255.)
    
    #x_processed = x
    
    h_fc01 = tf.multiply(x_processed, phase_errors)
    #h_fc01 = (h_fc01/255.*2)-1
    
    
    x_image = tf.reshape(h_fc01, [-1, n, n, 1])
    
    #residual_layers= deep_residual_ista_speckling_current(x_image,nums,keep_prob)
    
    residual_layers= deep_residual_ista_mix_speckling(x_image,nums,keep_prob)
    
    #residual_layers = [i * 255. for i in residual_layers]

    residual_vectors= layer_vectors(residual_layers,nums)

    
    return x,residual_layers,residual_vectors,h_fc01

def setup_model_autofocus_phase(x, keep_prob,keep_prob_2,phase_errors,nums):
    
    
    n= x.get_shape()[2].value
    
    dft_matrix= tf.constant(dft(n)/np.sqrt(n), dtype=tf.complex64)
    
    x = tf.cast(x, dtype=tf.complex64)
    
    dft_x=tf.tensordot(x,dft_matrix,axes=[[2],[0]])
    
    aa=tf.cast(tf.multiply(np.pi,phase_errors), dtype=tf.complex64)
    
    phase_error_amp= tf.exp(-2j*aa)
    
    phase_error_matrix = tf.matrix_diag(phase_error_amp)
    
    phase_error_matrix = tf.cast(phase_error_matrix, dtype=tf.complex64)
    
    dft_y=tf.multiply(dft_x,phase_error_matrix)
    
    
    W_fc01 = weight_variable([n, n])
    
    W_fc02 = weight_variable([n,n])
    
    W = tf.complex(W_fc01,W_fc02)

    
    h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y, W ,axes=[[2],[0]])))
    
    
    
    x_image = tf.reshape(h_fc01, [-1, n, n, 1])
    
    residual_layers,y= deep_residual_ista_phase(x_image,nums,keep_prob)

    residual_vectors= layer_vectors(residual_layers,nums)

    
    return x,residual_layers,residual_vectors,h_fc01,y




def deep_residual_ista_speckling_current(data,n_nodes,keep_prob):
   W_conv2= {}
   
   b_conv2= {}

   h_conv2= {}
   
   h_conv2_norm= {}

   x_im=data
   
   W_conv1 = weight_variable([3, 3, 1, 64])
   b_conv1 = bias_variable([64])

   h_conv1 = tf.nn.relu(conv2d_same(x_im, W_conv1) + b_conv1)
   
   x_m=h_conv1
   
   for i in range(6):
        
        # Convolutional layer 2
        W_conv2[i] = weight_variable([3, 3, 64, 64])
        b_conv2[i] = bias_variable([64])
        
        h_conv2[i] = conv2d_same(x_m, W_conv2[i]) + b_conv2[i]
        h_conv2_norm[i] = tf.nn.lrn(h_conv2[i], 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        
        x_m=tf.nn.relu(h_conv2_norm[i])
        
   W_conv3 = weight_variable([3, 3, 64, 1])
   b_conv3 = bias_variable([1])
   
   h = (conv2d_same(x_m, W_conv3) + b_conv3)
   
   h = h+ 1e-7
   
   h_conv3 = tf.add(data,h)
   #h_conv3 = tf.nn.sigmoid(tf.div(data,h))

   return [h_conv3*255.]


def deep_residual_conv_add(data,n_nodes,keep_prob):
   W_conv2= {}
   
   b_conv2= {}

   h_conv2= {}
   
   h_conv2_norm= {}

   x_im=data
   
   W_conv1 = weight_variable([3, 3, 1, 64])
   b_conv1 = bias_variable([64])

   h_conv1 = tf.nn.relu(conv2d_same(x_im, W_conv1) + b_conv1)
   
   x_m=h_conv1
   
   for i in range(6):
        
        # Convolutional layer 2
        W_conv2[i] = weight_variable([3, 3, 64, 64])
        b_conv2[i] = bias_variable([64])
        
        h_conv2[i] = tf.nn.relu(conv2d_same(x_m, W_conv2[i]) + b_conv2[i])
        h_conv2_norm[i] = tf.nn.lrn(h_conv2[i], 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        
        x_m=(h_conv2_norm[i])
        
   W_conv3 = weight_variable([3, 3, 64, 1])
   b_conv3 = bias_variable([1])
   
   h = (conv2d_same(x_m, W_conv3) + b_conv3)
   
   #h_conv3 = tf.add(data,h)
   h_conv3 = tf.add(data,h)

   return [h_conv3]

def deep_residual_ista_divide(data,n_nodes,keep_prob):
   W_conv1= {}
   W_conv2= {}
   W_conv3= {}
   W_conv4= {}
   W_conv5= {}
   W_conv6= {}
   
   
   b_conv1= {}
   b_conv2= {}
   b_conv3= {}
   b_conv4= {}
   b_conv5= {}
   b_conv6= {}
   
   
   h_conv1= {}
   h_conv2= {}
   h_conv3= {}
   h_conv3_drop= {}
   h_conv4= {}
   h_conv5= {}
   h_conv6= {}

   x_im=data
   
   
   
   for i in range(n_nodes):
        W_conv1[i] = weight_variable([3, 3, 1, 32])
        b_conv1[i] = bias_variable([32])
        
        h_conv1[i] = (conv2d_same(x_im, W_conv1[i]) + b_conv1[i])
        
        # Convolutional layer 2
        W_conv2[i] = weight_variable([3, 3, 32, 32])
        b_conv2[i] = bias_variable([32])
        
        h_conv2[i] = tf.nn.relu(conv2d_same(h_conv1[i], W_conv2[i]) + b_conv2[i])
        
        W_conv3[i] = weight_variable([3, 3, 32, 32])
        b_conv3[i] = bias_variable([32])
        
        h_conv3[i] = tf.nn.sigmoid(conv2d_same(h_conv2[i], W_conv3[i]) + b_conv3[i])
        
        h_conv3_drop[i]=tf.nn.dropout(h_conv3[i], keep_prob)
        
        W_conv4[i] = weight_variable([11, 11, 32, 64])
        b_conv4[i] = bias_variable([64])
        
        h_conv4[i] = tf.nn.relu(conv2d_same(h_conv3_drop[i], W_conv4[i]) + b_conv4[i])
        
        W_conv5[i] = weight_variable([1, 1, 64, 32])
        b_conv5[i] = bias_variable([32])
        
        h_conv5[i] = (conv2d_same(h_conv4[i], W_conv5[i]) + b_conv5[i])
        
        W_conv6[i] = weight_variable([7, 7, 32, 1])
        b_conv6[i] = bias_variable([1])
        
        #if i%2==0 :
        
        h_conv6[i] = tf.div(x_im,(conv2d_same(h_conv5[i], W_conv6[i]) + b_conv6[i])) 
        
        #else:
        #    h_conv6[i] = (conv2d(h_conv5[i], W_conv6[i]) + b_conv6[i])#+ x_im
            
        x_im= h_conv6[i]#
        

   return h_conv6


def deep_residual_ista_phase(data,n_nodes,keep_prob):
   W_conv1= {}
   W_conv2= {}
   W_conv3= {}
   W_conv4= {}
   W_conv5= {}
   W_conv6= {}
   
   
   b_conv1= {}
   b_conv2= {}
   b_conv3= {}
   b_conv4= {}
   b_conv5= {}
   b_conv6= {}
   
   
   h_conv1= {}
   h_conv2= {}
   h_conv3= {}
   h_conv3_drop= {}
   h_conv4= {}
   h_conv5= {}
   h_conv6= {}

   x_im=data
   
   
   
   for i in range(n_nodes):
        W_conv1[i] = weight_variable([3, 3, 1, 32])
        b_conv1[i] = bias_variable([32])
        
        h_conv1[i] = (conv2d_same(x_im, W_conv1[i]) + b_conv1[i])
        
        # Convolutional layer 2
        W_conv2[i] = weight_variable([3, 3, 32, 32])
        b_conv2[i] = bias_variable([32])
        
        h_conv2[i] = tf.nn.relu(conv2d_same(h_conv1[i], W_conv2[i]) + b_conv2[i])
        
        W_conv3[i] = weight_variable([3, 3, 32, 32])
        b_conv3[i] = bias_variable([32])
        
        h_conv3[i] = tf.nn.sigmoid(conv2d_same(h_conv2[i], W_conv3[i]) + b_conv3[i])
        
        h_conv3_drop[i]=tf.nn.dropout(h_conv3[i], keep_prob)
        
        W_conv4[i] = weight_variable([11, 11, 32, 64])
        b_conv4[i] = bias_variable([64])
        
        h_conv4[i] = tf.nn.relu(conv2d_same(h_conv3_drop[i], W_conv4[i]) + b_conv4[i])
        
        W_conv5[i] = weight_variable([1, 1, 64, 32])
        b_conv5[i] = bias_variable([32])
        
        h_conv5[i] = (conv2d_same(h_conv4[i], W_conv5[i]) + b_conv5[i])
        
        W_conv6[i] = weight_variable([7, 7, 32, 1])
        b_conv6[i] = bias_variable([1])
        
        #if i%2==0 :
        
        h_conv6[i] = (conv2d_same(h_conv5[i], W_conv6[i]) + b_conv6[i])+ x_im
        
        
        
        #else:
        #    h_conv6[i] = (conv2d(h_conv5[i], W_conv6[i]) + b_conv6[i])#+ x_im
            
        x_im= h_conv6[i]
        
   dft_matrix= tf.constant(dft(image_s)/np.sqrt(image_s), dtype=tf.complex64)
    
   x_im = tf.cast(tf.reshape(x_im,[-1,image_s,image_s]), dtype=tf.complex64)
    
   dft_xim=tf.tensordot(x_im,dft_matrix,axes=[[2],[0]])
        
   d_1 = weight_variable([image_s, 1])
   d_2 = weight_variable([image_s, 1])
   
   d= tf.complex(d_1,d_2)
    
   dft_a=tf.nn.tanh(tf.angle(tf.tensordot(dft_xim,d,axes=[[2],[0]]))/np.pi)
   
   dft_a= tf.reshape(dft_a,[-1,image_s])

   return h_conv6, dft_a


def deep_residual_ista_mix(data,n_nodes,keep_prob):
   W_conv1= {}
   W_conv2= {}
   W_conv3= {}
   W_conv4= {}
   W_conv5= {}
   W_conv6= {}
   
   
   b_conv1= {}
   b_conv2= {}
   b_conv3= {}
   b_conv4= {}
   b_conv5= {}
   b_conv6= {}
   
   
   h_conv1= {}
   h_conv2= {}
   h_conv3= {}
   h_conv3_drop= {}
   h_conv4= {}
   h_conv5= {}
   h_conv6= {}

   x_im=data
   
   
   
   for i in range(n_nodes):
        W_conv1[i] = weight_variable([3, 3, 1, 32])
        b_conv1[i] = bias_variable([32])
        
        h_conv1[i] = (conv2d_same(x_im, W_conv1[i]) + b_conv1[i])
        
        # Convolutional layer 2
        W_conv2[i] = weight_variable([3, 3, 32, 32])
        b_conv2[i] = bias_variable([32])
        
        h_conv2[i] = tf.nn.relu(conv2d_same(h_conv1[i], W_conv2[i]) + b_conv2[i])
        
        W_conv3[i] = weight_variable([3, 3, 32, 32])
        b_conv3[i] = bias_variable([32])
        
        h_conv3[i] = tf.nn.sigmoid(conv2d_same(h_conv2[i], W_conv3[i]) + b_conv3[i])
        
        h_conv3_drop[i]=tf.nn.dropout(h_conv3[i], keep_prob)
        
        W_conv4[i] = weight_variable([11, 11, 32, 64])
        b_conv4[i] = bias_variable([64])
        
        h_conv4[i] = tf.nn.relu(conv2d_same(h_conv3_drop[i], W_conv4[i]) + b_conv4[i])
        
        W_conv5[i] = weight_variable([1, 1, 64, 32])
        b_conv5[i] = bias_variable([32])
        
        h_conv5[i] = (conv2d_same(h_conv4[i], W_conv5[i]) + b_conv5[i])
        
        W_conv6[i] = weight_variable([7, 7, 32, 1])
        b_conv6[i] = bias_variable([1])
        
        #if i%2==0 :
        
        h_conv6[i] =( (conv2d_same(h_conv5[i], W_conv6[i]) + b_conv6[i])+ x_im)
        #else:
        #    h_conv6[i] = (conv2d(h_conv5[i], W_conv6[i]) + b_conv6[i])#+ x_im
            
        x_im= h_conv6[i]

   return h_conv6


def deep_residual_ista_mix_speckling(data,n_nodes,keep_prob):
   W_conv1= {}
   W_conv2= {}
   W_conv3= {}
   W_conv4= {}
   W_conv5= {}
   W_conv6= {}
   
   
   b_conv1= {}
   b_conv2= {}
   b_conv3= {}
   b_conv4= {}
   b_conv5= {}
   b_conv6= {}
   
   
   h_conv1= {}
   h_conv2= {}
   h_conv3= {}
   h_conv3_drop= {}
   h_conv4= {}
   h_conv5= {}
   h_conv6= {}

   x_im=data
   
   
   
   for i in range(n_nodes):
        W_conv1[i] = weight_variable([3, 3, 1, 64])
        b_conv1[i] = bias_variable([64])
        
        h_conv1[i] = (conv2d_same(x_im, W_conv1[i]) + b_conv1[i])
        h_conv1[i] = tf.nn.relu(h_conv1[i])
        
        # Convolutional layer 2
        W_conv2[i] = weight_variable([3, 3, 64, 64])
        b_conv2[i] = bias_variable([64])
        
        h_conv2[i] = tf.nn.lrn((conv2d_same(h_conv1[i], W_conv2[i]) + b_conv2[i]),4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        
        h_conv2[i] = tf.nn.relu(h_conv2[i])
        
        W_conv3[i] = weight_variable([3, 3, 64, 64])
        b_conv3[i] = bias_variable([64])
        
        h_conv3[i] = tf.nn.lrn((conv2d_same(h_conv2[i], W_conv3[i]) + b_conv3[i]),4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        
        h_conv3[i] = tf.nn.sigmoid(h_conv3[i])
        h_conv3_drop[i]=tf.nn.dropout(h_conv3[i], keep_prob)
        
        W_conv4[i] = weight_variable([11, 11, 64, 64])
        b_conv4[i] = bias_variable([64])
        
        h_conv4[i] = tf.nn.lrn((conv2d_same(h_conv3_drop[i], W_conv4[i]) + b_conv4[i]),4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        h_conv4[i] = tf.nn.relu(h_conv4[i])
        
        
        W_conv5[i] = weight_variable([1, 1, 64, 64])
        b_conv5[i] = bias_variable([64])
        
        h_conv5[i] = tf.nn.lrn((conv2d_same(h_conv4[i], W_conv5[i]) + b_conv5[i]),4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        
        h_conv5[i] = tf.nn.relu(h_conv5[i])
        
        W_conv6[i] = weight_variable([7, 7, 64, 1])
        b_conv6[i] = bias_variable([1])
        
        #if i%2==0 :
        
        h_conv6[i] =( (conv2d_same(h_conv5[i], W_conv6[i]) + b_conv6[i])+ x_im)
        #else:
        #    h_conv6[i] = (conv2d(h_conv5[i], W_conv6[i]) + b_conv6[i])#+ x_im
            
        
        x_im = h_conv6[i]

   return h_conv6


def deep_residual_ista_mix_complex(data,n_nodes,keep_prob):
    
   W_conv2= {}
   W_conv2_complex= {}
   
   b_conv2= {}
   b_conv2_complex= {}

   h_conv2= {}
   
   n= data.get_shape()[2].value
   
   layer_num=[128,64,32,64,128]
   
   data= tf.reshape(data,[-1,n,n])

   x_m=data
   
   
   for i in range(len(layer_num)):
        if i ==0:
            W_conv2[i] = weight_variable([256,layer_num[i]])
            b_conv2[i] = bias_variable([layer_num[i]])
            
            W_conv2_complex[i] = weight_variable([256,layer_num[i]])
            b_conv2_complex[i] = bias_variable([layer_num[i]])
            
            
            
        else:
            
            W_conv2[i] = weight_variable([layer_num[i-1],layer_num[i]])
            b_conv2[i] = bias_variable([layer_num[i]])
            
            W_conv2_complex[i] = weight_variable([layer_num[i-1],layer_num[i]])
            b_conv2_complex[i] = bias_variable([layer_num[i]])
            
            
        h_conv2[i] = tf.tensordot(x_m,tf.complex(W_conv2[i],W_conv2_complex[i]),axes=[[2],[0]]) + tf.complex(b_conv2[i],b_conv2_complex[i])
            
        x_m=tf.complex(tf.nn.leaky_relu(tf.real(h_conv2[i])),tf.nn.leaky_relu(tf.imag(h_conv2[i])))
        
        
   W_conv3_real = weight_variable([layer_num[-1], image_s])
   W_conv3_complex = weight_variable([layer_num[-1], image_s])
   
   b_conv3_real = bias_variable([image_s])
   b_conv3_complex = bias_variable([image_s])
   
   W_conv3=tf.complex(W_conv3_real, W_conv3_complex)
   b_conv3=tf.complex(b_conv3_real, b_conv3_complex)
   
   h = (tf.tensordot(x_m, W_conv3,axes=[[2],[0]]) + b_conv3)

   
    
   dft_matrix= tf.constant(dft(n)/np.sqrt(n), dtype=tf.complex64)
    
   
   h_conv3 =h
   
#   W_conv4_real = weight_variable([image_s, image_s])
#   W_conv4_complex = weight_variable([image_s, image_s])
#   
#   b_conv4_real = bias_variable([image_s])
#   b_conv4_complex = bias_variable([image_s])
#   
#   W_conv4=tf.complex(W_conv4_real, W_conv4_complex)
#   b_conv4=tf.complex(b_conv4_real, b_conv4_complex)
   
   
   
   
   h_conv6 = tf.nn.relu(tf.real(tf.tensordot(h_conv3, tf.transpose(dft_matrix, conjugate=True) ,axes=[[2],[0]])))

   return [h_conv6]



def setup_model_raw_cs(x, keep_prob,keep_prob_2,phase_errors, speck_errors,nums, batch_cs_matrix):
    
    
    n= x.get_shape()[2].value
    
#    x=x
#    
#    dft_m = (dft(n)/np.sqrt(n))
#    
#    dft_matrix= tf.constant(dft_m, dtype=tf.complex64)
#    
#    x_speckled=x
    
    #x_speckled = tf.cast(x_speckled, dtype=tf.complex64)
    
#    dft_x=tf.tensordot(x_speckled,dft_matrix,axes=[[2],[0]])
#    
#    aa=tf.cast(tf.multiply(np.pi,phase_errors), dtype=tf.complex64)
#    
#    phase_error_amp= tf.exp(-2j*aa)
#    
#    multiply = tf.constant([1,n])
#    
#    phase_error_matrix = tf.reshape(tf.tile(phase_error_amp, multiply), [-1, n, n])
#     
#    phase_error_matrix = tf.cast(phase_error_matrix, dtype=tf.complex64)
#    
#    dft_y=tf.multiply(dft_x,tf.transpose(phase_error_matrix,perm=[0,2,1]))
#
#    batch_cs_matrix = tf.cast(batch_cs_matrix, dtype=tf.complex64)
#    dft_y_cs=tf.multiply(dft_y,batch_cs_matrix)
    
    #batch_cs_matrix = weight_variable([n*n])
    
    #batch_cs_matrix_drop = tf.nn.dropout(batch_cs_matrix, keep_prob_2)
    
#    batch_cs_matrix_drop = tf.cast(batch_cs_matrix_drop, dtype=tf.complex64)
#    
#    dft_y= tf.reshape(dft_y,[-1,n*n])
#    
#    dft_y_cs=tf.einsum('ai,i->ai',dft_y,batch_cs_matrix_drop)
#    
#    dft_y_cs= tf.reshape(dft_y_cs,[-1,n,n])
    
    #dft_y_cs=tf.multiply(dft_y,tf.transpose(batch_cs_matrix,perm=[0,2,1]))
    
    
    
#    dft_y_cs_transpose= tf.transpose(dft_y_cs,perm=[1,0,2])
#    
#    W_fc01 = {}
#    W_fc02 = {}
#    
#    W={}
#    
#    for i in range(n):
#        
#        
#
#        W_fc01[i] = weight_variable([n, n])
#    
#        W_fc02[i] = weight_variable([n, n])
#    
#        W[i] = tf.complex(W_fc01[i],W_fc02[i])
#        
#        row_dft_y_cs= tf.gather_nd(dft_y_cs_transpose,[[i]])
#        
#        row_dft_y_cs= tf.reshape(row_dft_y_cs,[-1,1,n])
#        
#        if i ==0:
#            h_fc01 = tf.nn.relu(tf.real(tf.tensordot(row_dft_y_cs, W[i] ,axes=[[2],[0]])))
#        else:
#            tmp = tf.nn.relu(tf.real(tf.tensordot(row_dft_y_cs, W[i] ,axes=[[2],[0]])))
#            h_fc01 = tf.concat([h_fc01,tmp],1)
    
    
    W_fc01_real = weight_variable([n, n])
    
    W_fc01_imag = weight_variable([n, n])
    
    W_fc01 = tf.complex(W_fc01_real,W_fc01_imag)

    
    h_fc01 = (tf.tensordot(x, W_fc01 ,axes=[[2],[0]]))
    
    
    
    W_fc03_real = weight_variable([n, n])
    
    W_fc03_imag = weight_variable([n, n])
    
    W_fc03 = tf.complex(W_fc03_real,W_fc03_imag)

    
    h_fc03 = tf.einsum('kij,ij->kij', h_fc01, W_fc03)
    
    
    
    W_fc02_real = weight_variable([n, n])
    
    W_fc02_imag = weight_variable([n, n])
    
    W_fc02 = tf.complex(W_fc02_real,W_fc02_imag)

    
    h_fc02 = tf.transpose(tf.tensordot(tf.transpose(h_fc03,perm=[0,2,1]), W_fc02 ,axes=[[2],[0]]),perm=[0,2,1])

    
    W_fc04_real = weight_variable([n, n])
    
    W_fc04_imag = weight_variable([n, n])
    
    W_fc04 = tf.complex(W_fc04_real,W_fc04_imag)

    
    h_fc04 = tf.einsum('kij,ij->kij', h_fc02, W_fc04)
    
    W_fc05_real = weight_variable([n, n])
    
    W_fc05_imag = weight_variable([n, n])
    
    W_fc05 = tf.complex(W_fc05_real,W_fc05_imag)

    
    h_fc05 = tf.transpose(tf.tensordot(tf.transpose(h_fc04,perm=[0,2,1]), W_fc05 ,axes=[[2],[0]]),perm=[0,2,1])


    W_fc06_real = weight_variable([n, n])
    
    W_fc06_imag = weight_variable([n, n])
    
    W_fc06 = tf.complex(W_fc06_real,W_fc06_imag)

    
    h_fc06 = tf.einsum('kij,ij->kij', h_fc05, W_fc06)
    
    
    W_fc07_real = weight_variable([n, n])
    
    W_fc07_imag = weight_variable([n, n])
    
    W_fc07 = tf.complex(W_fc07_real,W_fc07_imag)

    
    h_fc07 = tf.nn.relu(tf.real(tf.tensordot(h_fc06, W_fc07 ,axes=[[2],[0]])))
    
    
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, W ,axes=[[2],[0]])))
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, tf.transpose(dft_matrix,conjugate=True) ,axes=[[2],[0]])))
    
    x_image = tf.reshape(h_fc07, [-1, n, n, 1])
    
    #residual_layers= deep_residual_conv_add(x_image,nums,keep_prob)
    
    residual_layers= deep_residual_ista_mix_speckling(x_image,nums,keep_prob)

    residual_vectors= layer_vectors(residual_layers,nums)

    
    return x,residual_layers,residual_vectors


def setup_model_raw_cs_2(x, keep_prob,keep_prob_2,phase_errors, speck_errors,nums, batch_cs_matrix):
    
    
    n= x.get_shape()[2].value
    
#    x=x
#    
#    dft_m = (dft(n)/np.sqrt(n))
#    
#    dft_matrix= tf.constant(dft_m, dtype=tf.complex64)
#    
#    x_speckled=x
    
    #x_speckled = tf.cast(x_speckled, dtype=tf.complex64)
    
#    dft_x=tf.tensordot(x_speckled,dft_matrix,axes=[[2],[0]])
#    
#    aa=tf.cast(tf.multiply(np.pi,phase_errors), dtype=tf.complex64)
#    
#    phase_error_amp= tf.exp(-2j*aa)
#    
#    multiply = tf.constant([1,n])
#    
#    phase_error_matrix = tf.reshape(tf.tile(phase_error_amp, multiply), [-1, n, n])
#     
#    phase_error_matrix = tf.cast(phase_error_matrix, dtype=tf.complex64)
#    
#    dft_y=tf.multiply(dft_x,tf.transpose(phase_error_matrix,perm=[0,2,1]))
#
#    batch_cs_matrix = tf.cast(batch_cs_matrix, dtype=tf.complex64)
#    dft_y_cs=tf.multiply(dft_y,batch_cs_matrix)
    
    #batch_cs_matrix = weight_variable([n*n])
    
    #batch_cs_matrix_drop = tf.nn.dropout(batch_cs_matrix, keep_prob_2)
    
#    batch_cs_matrix_drop = tf.cast(batch_cs_matrix_drop, dtype=tf.complex64)
#    
#    dft_y= tf.reshape(dft_y,[-1,n*n])
#    
#    dft_y_cs=tf.einsum('ai,i->ai',dft_y,batch_cs_matrix_drop)
#    
#    dft_y_cs= tf.reshape(dft_y_cs,[-1,n,n])
    
    #dft_y_cs=tf.multiply(dft_y,tf.transpose(batch_cs_matrix,perm=[0,2,1]))
    
    
    
#    dft_y_cs_transpose= tf.transpose(dft_y_cs,perm=[1,0,2])
#    
#    W_fc01 = {}
#    W_fc02 = {}
#    
#    W={}
#    
#    for i in range(n):
#        
#        
#
#        W_fc01[i] = weight_variable([n, n])
#    
#        W_fc02[i] = weight_variable([n, n])
#    
#        W[i] = tf.complex(W_fc01[i],W_fc02[i])
#        
#        row_dft_y_cs= tf.gather_nd(dft_y_cs_transpose,[[i]])
#        
#        row_dft_y_cs= tf.reshape(row_dft_y_cs,[-1,1,n])
#        
#        if i ==0:
#            h_fc01 = tf.nn.relu(tf.real(tf.tensordot(row_dft_y_cs, W[i] ,axes=[[2],[0]])))
#        else:
#            tmp = tf.nn.relu(tf.real(tf.tensordot(row_dft_y_cs, W[i] ,axes=[[2],[0]])))
#            h_fc01 = tf.concat([h_fc01,tmp],1)
    
    
    W_fc01_real = weight_variable([n, n])
    
    W_fc01_imag = weight_variable([n, n])
    
    W_fc01 = tf.complex(W_fc01_real,W_fc01_imag)

    
    #h_fc01 = (tf.tensordot(x, W_fc01 ,axes=[[2],[0]]))
    h_fc01 = tf.transpose(tf.tensordot(tf.transpose(x,perm=[0,2,1]), W_fc01 ,axes=[[2],[0]]),perm=[0,2,1])

    
    
    W_fc03_real = weight_variable([n, n])
    
    W_fc03_imag = weight_variable([n, n])
    
    W_fc03 = tf.complex(W_fc03_real,W_fc03_imag)

    
    h_fc03 = tf.einsum('kij,ij->kij', h_fc01, W_fc03)
    
    
    
    W_fc02_real = weight_variable([n, n])
    
    W_fc02_imag = weight_variable([n, n])
    
    W_fc02 = tf.complex(W_fc02_real,W_fc02_imag)

    h_fc02 = (tf.tensordot(h_fc03, W_fc02 ,axes=[[2],[0]]))
    
    #h_fc02 = tf.transpose(tf.tensordot(tf.transpose(h_fc03,perm=[0,2,1]), W_fc02 ,axes=[[2],[0]]),perm=[0,2,1])

    
    W_fc04_real = weight_variable([n, n])
    
    W_fc04_imag = weight_variable([n, n])
    
    W_fc04 = tf.complex(W_fc04_real,W_fc04_imag)

    
    h_fc04 = tf.einsum('kij,ij->kij', h_fc02, W_fc04)
    
    W_fc05_real = weight_variable([n, n])
    
    W_fc05_imag = weight_variable([n, n])
    
    W_fc05 = tf.complex(W_fc05_real,W_fc05_imag)

    h_fc05 = (tf.tensordot(h_fc04, W_fc05 ,axes=[[2],[0]]))
    #h_fc05 = tf.transpose(tf.tensordot(tf.transpose(h_fc04,perm=[0,2,1]), W_fc05 ,axes=[[2],[0]]),perm=[0,2,1])


    W_fc06_real = weight_variable([n, n])
    
    W_fc06_imag = weight_variable([n, n])
    
    W_fc06 = tf.complex(W_fc06_real,W_fc06_imag)

    
    h_fc06 = tf.einsum('kij,ij->kij', h_fc05, W_fc06)
    
    
    W_fc07_real = weight_variable([n, n])
    
    W_fc07_imag = weight_variable([n, n])
    
    W_fc07 = tf.complex(W_fc07_real,W_fc07_imag)

    h_fc07 = tf.nn.relu(tf.real(tf.transpose(tf.tensordot(tf.transpose(h_fc06,perm=[0,2,1]), W_fc07 ,axes=[[2],[0]]),perm=[0,2,1])))

    #h_fc07 = tf.nn.relu(tf.real(tf.tensordot(h_fc06, W_fc07 ,axes=[[2],[0]])))
    
    
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, W ,axes=[[2],[0]])))
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, tf.transpose(dft_matrix,conjugate=True) ,axes=[[2],[0]])))
    
    x_image = tf.reshape(h_fc07, [-1, n, n, 1])
    
    #residual_layers= deep_residual_conv_add(x_image,nums,keep_prob)
    
    residual_layers= deep_residual_ista_mix_speckling(x_image,nums,keep_prob)

    residual_vectors= layer_vectors(residual_layers,nums)

    
    return x,residual_layers,residual_vectors



def setup_model_raw_cs_3(x, keep_prob,keep_prob_2,phase_errors, speck_errors,nums, batch_cs_matrix):
    
    
    n= x.get_shape()[2].value
    
#    x=x
#    
    dft_m = (dft(n)/np.sqrt(n))
#    
    dft_matrix= tf.constant(dft_m, dtype=tf.complex64)
    
    inv_dft_matrix= tf.transpose(dft_matrix, conjugate=True)
#    
#    x_speckled=x
    
    #x_speckled = tf.cast(x_speckled, dtype=tf.complex64)
    
#    dft_x=tf.tensordot(x_speckled,dft_matrix,axes=[[2],[0]])
#    
#    aa=tf.cast(tf.multiply(np.pi,phase_errors), dtype=tf.complex64)
#    
#    phase_error_amp= tf.exp(-2j*aa)
#    
#    multiply = tf.constant([1,n])
#    
#    phase_error_matrix = tf.reshape(tf.tile(phase_error_amp, multiply), [-1, n, n])
#     
#    phase_error_matrix = tf.cast(phase_error_matrix, dtype=tf.complex64)
#    
#    dft_y=tf.multiply(dft_x,tf.transpose(phase_error_matrix,perm=[0,2,1]))
#
#    batch_cs_matrix = tf.cast(batch_cs_matrix, dtype=tf.complex64)
#    dft_y_cs=tf.multiply(dft_y,batch_cs_matrix)
    
    #batch_cs_matrix = weight_variable([n*n])
    
    #batch_cs_matrix_drop = tf.nn.dropout(batch_cs_matrix, keep_prob_2)
    
#    batch_cs_matrix_drop = tf.cast(batch_cs_matrix_drop, dtype=tf.complex64)
#    
#    dft_y= tf.reshape(dft_y,[-1,n*n])
#    
#    dft_y_cs=tf.einsum('ai,i->ai',dft_y,batch_cs_matrix_drop)
#    
#    dft_y_cs= tf.reshape(dft_y_cs,[-1,n,n])
    
    #dft_y_cs=tf.multiply(dft_y,tf.transpose(batch_cs_matrix,perm=[0,2,1]))
    
    
    
#    dft_y_cs_transpose= tf.transpose(dft_y_cs,perm=[1,0,2])
#    
#    W_fc01 = {}
#    W_fc02 = {}
#    
#    W={}
#    
#    for i in range(n):
#        
#        
#
#        W_fc01[i] = weight_variable([n, n])
#    
#        W_fc02[i] = weight_variable([n, n])
#    
#        W[i] = tf.complex(W_fc01[i],W_fc02[i])
#        
#        row_dft_y_cs= tf.gather_nd(dft_y_cs_transpose,[[i]])
#        
#        row_dft_y_cs= tf.reshape(row_dft_y_cs,[-1,1,n])
#        
#        if i ==0:
#            h_fc01 = tf.nn.relu(tf.real(tf.tensordot(row_dft_y_cs, W[i] ,axes=[[2],[0]])))
#        else:
#            tmp = tf.nn.relu(tf.real(tf.tensordot(row_dft_y_cs, W[i] ,axes=[[2],[0]])))
#            h_fc01 = tf.concat([h_fc01,tmp],1)
    
    
    W_fc01_real = weight_variable([n, n])
    
    W_fc01_imag = weight_variable([n, n])
    
    W_fc01 = tf.complex(W_fc01_real,W_fc01_imag)

    
    #h_fc01 = (tf.tensordot(x, W_fc01 ,axes=[[2],[0]]))
    h_fc01 = tf.transpose(tf.tensordot(tf.transpose(x,perm=[0,2,1]), dft_matrix ,axes=[[2],[0]]),perm=[0,2,1])

    
    
    W_fc03_real = weight_variable([n, n])
    
    W_fc03_imag = weight_variable([n, n])
    
    W_fc03 = tf.complex(W_fc03_real,W_fc03_imag)

    
    h_fc03 = tf.einsum('kij,ij->kij', h_fc01, W_fc03)
    
    
    
    W_fc02_real = weight_variable([n, n])
    
    W_fc02_imag = weight_variable([n, n])
    
    W_fc02 = tf.complex(W_fc02_real,W_fc02_imag)

    h_fc02 = (tf.tensordot(h_fc03, dft_matrix  ,axes=[[2],[0]]))
    
    #h_fc02 = tf.transpose(tf.tensordot(tf.transpose(h_fc03,perm=[0,2,1]), W_fc02 ,axes=[[2],[0]]),perm=[0,2,1])

    
    W_fc04_real = weight_variable([n, n])
    
    W_fc04_imag = weight_variable([n, n])
    
    W_fc04 = tf.complex(W_fc04_real,W_fc04_imag)

    
    h_fc04 = tf.einsum('kij,ij->kij', h_fc02, W_fc04)
    
    W_fc05_real = weight_variable([n, n])
    
    W_fc05_imag = weight_variable([n, n])
    
    W_fc05 = tf.complex(W_fc05_real,W_fc05_imag)

    h_fc05 = (tf.tensordot(h_fc04, inv_dft_matrix  ,axes=[[2],[0]]))
    #h_fc05 = tf.transpose(tf.tensordot(tf.transpose(h_fc04,perm=[0,2,1]), W_fc05 ,axes=[[2],[0]]),perm=[0,2,1])


    W_fc06_real = weight_variable([n, n])
    
    W_fc06_imag = weight_variable([n, n])
    
    W_fc06 = tf.complex(W_fc06_real,W_fc06_imag)

    
    h_fc06 = tf.einsum('kij,ij->kij', h_fc05, W_fc06)
    
    
    W_fc07_real = weight_variable([n, n])
    
    W_fc07_imag = weight_variable([n, n])
    
    W_fc07 = tf.complex(W_fc07_real,W_fc07_imag)

    h_fc07 = tf.nn.relu(tf.real(tf.transpose(tf.tensordot(tf.transpose(h_fc06,perm=[0,2,1]), inv_dft_matrix ,axes=[[2],[0]]),perm=[0,2,1])))

    #h_fc07 = tf.nn.relu(tf.real(tf.tensordot(h_fc06, W_fc07 ,axes=[[2],[0]])))
    
    
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, W ,axes=[[2],[0]])))
    
    #h_fc01 = tf.nn.relu(tf.real(tf.tensordot(dft_y_cs, tf.transpose(dft_matrix,conjugate=True) ,axes=[[2],[0]])))
    
    x_image = tf.reshape(h_fc07, [-1, n, n, 1])
    
    #residual_layers= deep_residual_conv_add(x_image,nums,keep_prob)
    
    residual_layers= deep_residual_ista_mix_speckling(x_image,nums,keep_prob)

    residual_vectors= layer_vectors(residual_layers,nums)

    
    return x,residual_layers,residual_vectors