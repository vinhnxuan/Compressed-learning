# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:36:11 2018

@author: vinh
"""

from __future__ import print_function

import numpy as np
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

learning = tf.placeholder('float')

learning2 = tf.placeholder('float')

num_measurement=300;


def inputTrainDistort(train_data):
  
    #Loop through training data. Reassemble 28x28 arrays and shift 1 pixel each direction.
    #Flatten new images and populate expanded X_shifted array with both old and new flattened images.
    X = train_data[:,1:]
    pieces = [X,X,X,X,X]
    X_shifted = np.concatenate(pieces, axis=0)

    for m in range(0,train_data.shape[0]-1):
        pixels = train_data[m,:]
        pixels = np.array(pixels, dtype='uint8')
        pixels = pixels.reshape(28,28)

        pixels_shift_rt1 = np.roll(pixels, 1)
        pixels_shift_rt1[:,0] = 0

        pixels_shift_lf1 = np.roll(pixels, -1)
        pixels_shift_lf1[:,27] = 0

        pixels_shift_dn1 = np.roll(pixels, 1, axis=0)
        pixels_shift_dn1[:,0] = 0

        pixels_shift_up1 = np.roll(pixels, -1, axis=0)
        pixels_shift_up1[:,27] = 0
        
        X_shifted[m] = pixels.reshape(1,784)
        X_shifted[m + 1*train_data.shape[0]] = pixels_shift_rt1.reshape(1,784)
        X_shifted[m + 2*train_data.shape[0]] = pixels_shift_lf1.reshape(1,784)
        X_shifted[m + 3*train_data.shape[0]] = pixels_shift_dn1.reshape(1,784)
        X_shifted[m + 4*train_data.shape[0]] = pixels_shift_up1.reshape(1,784)

    #Make a new y vector of 5 stacks of y (1 original plus 4 copies for 
    #translated X's).
    y = train_data[:,0]
    pieces_y = [y,y,y,y,y]
    y_shifted = np.concatenate(pieces_y, axis=0)
    y_shifted = y_shifted[:, None]
    
    #put the expanded and reflattened training set numpy array back together
    Train = np.concatenate((y_shifted, X_shifted), axis=1)
    return Train

def get_data_set(name="train", cifar=10):
    X = None
    Y = None
    L = None
    
    def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot
    
    folder_name = "cifar_10" if cifar == 10 else "cifar_100"

    f = open('data_set/'+folder_name+'/batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    if folder_name is "cifar_10":
        L = datadict['label_names']
    else:
        L = datadict['fine_label_names']

    if name is "train":
        if folder_name is "cifar_10":
            for i in range(5):
                f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
                datadict = pickle.load(f, encoding='latin1')
                f.close()
    
                _X = datadict["data"]
                _Y = datadict['labels']
    
                _X = np.array(_X, dtype=float) / 255.0
                _X = _X.reshape([-1, 3, 32, 32])
                _X = _X.transpose([0, 2, 3, 1])
                
                _X = np.mean(_X, axis=3)
            
                _X = _X.reshape(-1,32*32)
    
                if X is None:
                    X = _X
                    Y = _Y
                else:
                    X = np.concatenate((X, _X), axis=0)
                    Y = np.concatenate((Y, _Y), axis=0)
        else:
            f = open('./data_set/'+folder_name+'/train_batch', 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

    
            X = datadict["data"]
            
            Y = np.array(datadict['fine_labels'])
    
            X = np.array(X, dtype=float) / 255.0
            X = X.reshape([-1, 3, 32, 32])
            X = X.transpose([0, 2, 3, 1])
            X= np.mean(X, axis=3)
        
        
            X = X.reshape(-1, 32*32)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        X = datadict["data"]
        if folder_name is "cifar_10":
            Y = np.array(datadict['labels'])
        else:
            Y = np.array(datadict['fine_labels'])

        X = np.array(X, dtype=float) / 255.0
        X = X.reshape([-1, 3, 32, 32])
        X = X.transpose([0, 2, 3, 1])
        X= np.mean(X, axis=3)
        
        
        X = X.reshape(-1, 32*32)
    S = X.shape
    L = int(S[0]/batch_size)
    
    
    return X,Y,L

def load_mnist(batch_size, is_training=True, order=5):
    path = os.path.normpath("/home/ads/gw438/vinh/MNIST_data/")
    
    
    if is_training:
        
        fd = open(os.path.join(path, 'train-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
        
        trainX=trainX.reshape(60000,784)

        fd = open(os.path.join(path, 'train-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:60000] / 255.
        trY = trainY[:60000]

        valX = trainX[50000:, ] / 255.
        valY = trainY[50000:]

        num_tr_batch = 60000 // batch_size
        num_val_batch = 10000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
        
        teX=teX.reshape(10000,784)

        fd = open(os.path.join(path, 't10k-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch

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
  
def get_mmse(input_, logits):
  return tf.reduce_mean( (input_ - logits) ** 2)

def get_psnr(input_, logits):
    mse = tf.reduce_mean( (input_ - logits) ** 2, axis=1 )
    PIXEL_MAX = 1.0
    
    return tf.reduce_mean (20 * tf.log(PIXEL_MAX / tf.sqrt(mse))/tf.log(tf.constant(10, dtype=tf.float32)))
  
def layer_vectors(layers,n_nodes):
    
    layer_vecs= {}
    for i in range(n_nodes):
        layer_vecs[i]=tf.reshape(layers[i], [-1, 32*32])
    return layer_vecs


def residual_loss(data,vectors,n_nodes):
    _loss_2=0
    if n_nodes>1:
        for i in range(n_nodes-1):
            _loss_2=_loss_2+get_mmse(data,vectors[i])
    return _loss_2

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
        
        h_conv1[i] = (conv2d(x_im, W_conv1[i]) + b_conv1[i])
        
        # Convolutional layer 2
        W_conv2[i] = weight_variable([3, 3, 32, 32])
        b_conv2[i] = bias_variable([32])
        
        h_conv2[i] = tf.nn.relu(conv2d(h_conv1[i], W_conv2[i]) + b_conv2[i])
        
        W_conv3[i] = weight_variable([3, 3, 32, 32])
        b_conv3[i] = bias_variable([32])
        
        h_conv3[i] = tf.nn.sigmoid(conv2d(h_conv2[i], W_conv3[i]) + b_conv3[i])
        
        h_conv3_drop[i]=tf.nn.dropout(h_conv3[i], keep_prob)
        
        W_conv4[i] = weight_variable([11, 11, 32, 64])
        b_conv4[i] = bias_variable([64])
        
        h_conv4[i] = tf.nn.relu(conv2d(h_conv3_drop[i], W_conv4[i]) + b_conv4[i])
        
        W_conv5[i] = weight_variable([1, 1, 64, 32])
        b_conv5[i] = bias_variable([32])
        
        h_conv5[i] = (conv2d(h_conv4[i], W_conv5[i]) + b_conv5[i])
        
        W_conv6[i] = weight_variable([7, 7, 32, 1])
        b_conv6[i] = bias_variable([1])
        
        #if i%2==0 :
        
        h_conv6[i] = (conv2d(h_conv5[i], W_conv6[i]) + b_conv6[i])+ x_im
        #else:
        #    h_conv6[i] = (conv2d(h_conv5[i], W_conv6[i]) + b_conv6[i])#+ x_im
            
        x_im= h_conv6[i]

   return h_conv6    


batch_size=50
trX, trY, num_tr_batch = get_data_set(name="train", cifar=10)
valX, valY, num_val_batch = get_data_set(name="test", cifar=10)
valX2, valY2, num_val_batch2 = get_data_set(name="test", cifar=100)


trY = trY[:num_tr_batch * batch_size].reshape((-1, 1))


valY = valY[:num_val_batch * batch_size].reshape((-1, 1))

valY2 = valY2[:num_val_batch2 * batch_size].reshape((-1, 1))


# Input layer
x  = tf.placeholder(tf.float32, [None, 1024], name='x')
y_ = tf.placeholder(tf.int32, [None, 1],  name='y_')

noise  = tf.placeholder(tf.float32, [None, num_measurement], name='n')

keep_prob  = tf.placeholder(tf.float32)

y1 = tf.one_hot(y_, depth=10, axis=1, dtype=tf.float32)
y1 = tf.reshape(y1,(-1, 10))

W_fc01 = weight_variable([1024, num_measurement])
#b_fc01 = bias_variable([num_measurement])


W_m= tf.transpose(tf.transpose(W_fc01))

W_norm= tf.norm(W_m,ord='euclidean',axis=1,keepdims=True)
norm_mat=tf.divide(W_m,W_norm)
coherence=tf.matmul(norm_mat,tf.transpose(norm_mat))-tf.diag(tf.ones([32*32]))


c_est=tf.reduce_max((tf.abs(coherence)))

h_fc01 = (tf.matmul(x, W_fc01))

h_fc01_noisy = h_fc01+ noise


W_fc02 = weight_variable([num_measurement, 1024])
b_fc02 = bias_variable([1024])

h_fc02 = tf.nn.relu(tf.matmul(h_fc01_noisy, W_fc02)+b_fc02)
#
x_02 = tf.reshape(h_fc02, [-1, 32, 32, 1])

W_fc03 = weight_variable([num_measurement, 1024*16])
b_fc03 = bias_variable([1024*16])

h_fc03 = tf.nn.relu(tf.matmul(h_fc01_noisy, W_fc03)+b_fc03)


x_03 = tf.reshape(h_fc03, [-1, 32, 32, 16])


W_conv10 = weight_variable([11 ,11 , 16, 64])
b_conv10 = bias_variable([64])

h_conv10 = (conv2d(x_03, W_conv10) + b_conv10)

W_conv11 = weight_variable([1, 1, 64, 64])
b_conv11 = bias_variable([64])

h_conv11 = tf.nn.relu(conv2d(h_conv10, W_conv11) + b_conv11)

W_conv12 = weight_variable([7, 7, 64, 1])
b_conv12 = bias_variable([1])

h_conv12 = (conv2d(h_conv11, W_conv12) + b_conv12) 


x_image = h_conv12 + x_02 

h_conv03_vector= tf.reshape(x_image, [-1, 32*32])


nums=2

residual_layers= deep_residual_ista_mix(x_image,nums,keep_prob)

residual_vectors= layer_vectors(residual_layers,nums)




MM= tf.reduce_mean(tf.abs(W_fc01))+ tf.reduce_mean(tf.abs(W_fc02))  + tf.reduce_mean(tf.abs(b_fc02)) 
MM2= tf.reduce_mean(tf.abs(W_fc01))#tf.reduce_mean(tf.abs(W_conv1))+tf.reduce_mean(tf.abs(W_conv2))+tf.reduce_mean(tf.abs(W_conv3))+tf.reduce_mean(tf.abs(W_conv4))+tf.reduce_mean(tf.abs(W_conv5))+tf.reduce_mean(tf.abs(W_conv6))
#MM3= tf.reduce_mean(tf.abs(b_conv1))+tf.reduce_mean(tf.abs(b_conv2))+tf.reduce_mean(tf.abs(b_conv3))+tf.reduce_mean(tf.abs(b_conv4))+tf.reduce_mean(tf.abs(b_conv5))+tf.reduce_mean(tf.abs(b_conv6))

MM3=tf.sqrt(tf.reduce_mean(tf.square(tf.image.total_variation(residual_layers[nums-1]))))

ratio=0.002
_loss_2= residual_loss(x,residual_vectors,nums)+ get_mmse(x,h_conv03_vector)

#_loss_2=get_mmse(x,h_fc02) +get_mmse(x,h_conv04_vector)+get_mmse(x,h_conv92_vector)

_loss = get_mmse(x,residual_vectors[nums-1]) +0.1*_loss_2

_psnr = get_psnr(x,residual_vectors[nums-1])

cos=_loss+ ratio*(MM)#+ 0.00005*c_est#+0.0001*(tf.reduce_mean(-tf.log(prob)))

#cos2=tf.reduce_mean(-tf.log(prob)+tf.log(1-prob))


# Training algorithm
optimizer_1 = tf.train.AdamOptimizer(learning).minimize(cos)
#optimizer_2 = tf.train.AdamOptimizer(learning2).minimize(cos2, var_list=[W_conv07,W_conv08,W_conv09,W_fc10, b_conv07,b_conv08,b_conv09,b_fc10])


learning_rate=3e-4
learning_rate_2=1e-5

learning_rate0=learning_rate
checking=False
acc_max=0
acc_max2=0
acc_noise=0

ind_max=0



test_period=550
update_period=int(mnist.train.num_examples/batch_size)


# Training steps
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  max_epoch=2000
  for epoch in range(max_epoch):
    kk = np.arange(50000)
    np.random.shuffle(kk)
    loss=0
    if (epoch-ind_max >30) or checking:
                       
        if not checking:
            #ind_max= step
            checking=True
            learning_rate=min(learning_rate0,learning_rate)
        else:
            learning_rate=learning_rate*0.993
                
        optimizer=optimizer_1
        print('Phase 2')
        print(learning_rate)
    else:
        print('Phase 1')
        optimizer=optimizer_1
        learning_rate=learning_rate*0.993
        
    loss_reg=0
    loss_reg2=0
    loss_reg3=0
        
    for step in range(num_tr_batch):
        start = step * batch_size
        end = start + batch_size
        noise_data=np.zeros((batch_size,num_measurement)).astype(np.float32)
    #batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, c, W, reg, reg_M2, reg_M3 = sess.run([optimizer, _loss, W_fc01, MM, MM2, MM3], feed_dict={x: trX[kk[start:end]], y_: trY[kk[start:end]], learning:learning_rate,learning2:learning_rate_2, noise:noise_data, keep_prob:0.7})
        #acc = sess.run([accuracy], feed_dict={x: valX[start:end], y_: valY[start:end], keep_prob: 1.0,keep_prob2: 1.0})
        #_=sess.run([optimizer_2],feed_dict={x: trX[kk[start:end]], y_: trY[kk[start:end]], learning:learning_rate,learning2:learning_rate_2, noise:noise_data, keep_prob:0.5})
        #_=sess.run([optimizer_2],feed_dict={x: trX[kk[start:end]], y_: trY[kk[start:end]], learning:learning_rate,learning2:learning_rate_2, noise:noise_data, keep_prob:0.5})
        
        
        loss=loss+c
        loss_reg3=reg_M3
    loss_reg=reg
    loss_reg2=reg_M2
    
    
    val_acc = 0
    
    snr=20
    
    noise_data=np.zeros((10000,num_measurement)).astype(np.float32)
    
    acc_noiseless,coh = sess.run([_psnr,c_est], feed_dict={x: valX, y_: valY, noise:noise_data, keep_prob:1.0})
    
    acc_2,coh = sess.run([_psnr,c_est], feed_dict={x: valX2, y_: valY2, noise:noise_data, keep_prob:1.0})

    
    
    meas=np.matmul(valX,W)
    
    noise_data=np.random.normal(0.0,0.1,(10000,num_measurement)).astype(np.float32)
    
    for kk in range(10000):
        mea=meas[kk,]
        s_power=np.mean(np.square(mea))
        #s_power=np.mean(np.abs(mea))
        #n_sigma=(10**(-snr/20))*s_power
        
        noi=noise_data[kk,]
        n_power=np.mean(np.square(noi))
        ra=(s_power/n_power)*(10**(-snr/10))
        noise_data[kk,]=noise_data[kk,]*np.sqrt(ra)
    
    acc,coh = sess.run([_psnr,c_est], feed_dict={x: valX, y_: valY, noise:noise_data, keep_prob:1.0})
    #for i in range(num_val_batch):
    #    start = i * batch_size
    #    end = start + batch_size
    
        #acc=0
    val_acc = acc_noiseless
    print(epoch, val_acc, loss,loss_reg, loss_reg2, loss_reg3, ind_max, acc_max, acc_noise, acc_max2, coh)

    if epoch ==0 or acc_max<val_acc:
        acc_max=val_acc
        ind_max=epoch
        acc_noise=acc
        acc_max2=acc_2
    
    









