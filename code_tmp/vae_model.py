import tensorflow as tf
import numpy as np

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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding='SAME')

def conv7d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 8, 8, 1],padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')

def setup_model_sda(noisy_cs_measurements, image_shape):
    
    num_measurement= noisy_cs_measurements.get_shape()[1].value
    
    layer_width=700
    
    var_list = []
    
    W_fc03 = variable_with_weight_decay('weights003', shape=[num_measurement, layer_width], stddev=5e-2, wd=None)
    b_fc03 = variable_on_cpu('biases003', [layer_width], tf.constant_initializer(0.0))

    h_fc03 = tf.nn.relu(tf.matmul(noisy_cs_measurements, W_fc03)+b_fc03)
    
    var_list.append(W_fc03)
    var_list.append(b_fc03)
    
    W_fc04 = variable_with_weight_decay('weights004', shape=[layer_width, layer_width], stddev=5e-2, wd=None)
    b_fc04 = variable_on_cpu('biases004', [layer_width], tf.constant_initializer(0.0))
    
    h_fc04 = tf.nn.relu(tf.matmul(h_fc03, W_fc04)+b_fc04)
    
    var_list.append(W_fc04)
    var_list.append(b_fc04)
    
    W_fc05 = variable_with_weight_decay('weights005', shape=[layer_width, image_shape], stddev=5e-2, wd=None)
    b_fc05 = variable_on_cpu('biases005', [image_shape], tf.constant_initializer(0.0))
    
    y_est = tf.nn.sigmoid(tf.matmul(h_fc04, W_fc05) + b_fc05)
    
    var_list.append(W_fc05)
    var_list.append(b_fc05)
    
    MM= tf.reduce_mean(tf.abs(W_fc05)) + tf.reduce_mean(tf.abs(b_fc05))
    
    return y_est, MM, var_list

def setup_model_dr2net(noisy_cs_measurements, image_shape):
    
    num_measurement= noisy_cs_measurements.get_shape()[1].value
    
    print(num_measurement)
    
    width=int(np.sqrt(image_shape))
    
    var_list =[]
    
    W_fc02 = weight_variable([num_measurement, image_shape])
    b_fc02 = bias_variable([image_shape])
    
    h_fc02 = tf.nn.relu(tf.matmul(noisy_cs_measurements, W_fc02)+b_fc02)
    
    
    var_list.append(W_fc02)
    var_list.append(b_fc02)
    
    x_image = tf.reshape(h_fc02, [-1, width, width, 1])
    
    #x_image = gaussian_noise_layer(x_image, .2)
    
    
    #x_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x_image)
    
    
    # Convolutional layer 2
    W_conv1 = weight_variable([11, 11, 1, 64])
    b_conv1 = bias_variable([64])
    
    var_list.append(W_conv1)
    var_list.append(b_conv1)
    
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_conv1_norm = tf.nn.lrn(h_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    
    W_conv2 = weight_variable([1, 1, 64, 32])
    b_conv2 = bias_variable([32])
    
    var_list.append(W_conv2)
    var_list.append(b_conv2)
    
    h_conv2 = tf.nn.relu(conv2d(h_conv1_norm, W_conv2) + b_conv2)
    h_conv2_norm = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    
    W_conv3 = weight_variable([7, 7, 32, 1])
    b_conv3 = bias_variable([1])
    
    var_list.append(W_conv3)
    var_list.append(b_conv3)
    
    h_conv3 = (conv2d(h_conv2_norm, W_conv3) + b_conv3)
    
    h_conv3_add= x_image + h_conv3
    
    
        
    W_conv4 = weight_variable([11, 11, 1, 64])
    b_conv4 = bias_variable([64])
    
    var_list.append(W_conv4)
    var_list.append(b_conv4)
    
    h_conv4 = tf.nn.relu(conv2d(h_conv3_add, W_conv4) + b_conv4)
    h_conv4_norm = tf.nn.lrn(h_conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    
    W_conv5 = weight_variable([1, 1, 64, 32])
    b_conv5 = bias_variable([32])
    
    var_list.append(W_conv5)
    var_list.append(b_conv5)
    
    h_conv5 = tf.nn.relu(conv2d(h_conv4_norm, W_conv5) + b_conv5)
    h_conv5_norm = tf.nn.lrn(h_conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    
    W_conv6 = weight_variable([7, 7, 32, 1])
    b_conv6 = bias_variable([1])
    
    var_list.append(W_conv6)
    var_list.append(b_conv6)
    
    y = (conv2d(h_conv5_norm, W_conv6) + b_conv6) + h_conv3_add
    
    y_est =  tf.reshape(y, [-1, 32*32])
    
    MM= tf.reduce_mean(tf.abs(W_fc02)) + tf.reduce_mean(tf.abs(b_fc02)) 


    return y_est, MM,var_list