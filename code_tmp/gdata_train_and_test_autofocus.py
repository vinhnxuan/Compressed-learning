# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:36:11 2018

@author: vinh
"""

from __future__ import print_function

import numpy as np

import tensorflow as tf

import mstar_read_dataset as input_set

import mstar_model as model

import os

import matplotlib.pyplot as plt

_SAVE_PATH = "./tensorboard/MSTAR/"


os.environ["CUDA_VISIBLE_DEVICES"]="1"

num_measurement=8;
batch_size=25
keep_prob_value_1=0.5
keep_prob_value_2=0.5
learning_rate_init=1e-3
learning_decay=0.996

image_size=256

trX, trY = input_set.get_google_data('train',width=image_size, height=image_size, crop_size=image_size)

valX, valY = input_set.get_google_data('test',width=image_size, height=image_size, crop_size=image_size)

PIXEL_MAX1= np.amax(trX)
PIXEL_MAX2= np.amax(valX)
PIXEL_MAX=np.maximum(PIXEL_MAX1, PIXEL_MAX2)

print(PIXEL_MAX)

# regularization factor
beta=100


def save_to():
    export_file_txt="tmp/train_txt_"+str(num_measurement)+".txt"
    export_file_result = "tmp/train_txt_"+str(num_measurement)+".csv"


    fd_txt = open(export_file_txt, 'w')
    fd_txt.write('End to end CNN\n')
    fd_result = open(export_file_result, 'w')
    fd_result.write('epoch,train_acc,loss,val_acc,val_max\n')
    return(fd_txt, fd_result)
    
def compute_mcoherence(w):
    num= w.shape[0]
    for ii2 in range(num):
        gaus_vector=w[ii2,0:]
        mag = np.sqrt(np.dot(gaus_vector,gaus_vector))

        if mag>0:
            w[ii2,0:]=gaus_vector/mag
            
    
            
    coherence=np.matmul(w, np.transpose(w))-np.diag(np.ones(num))
    muy=np.amax(np.abs(coherence))
    
    return muy

def coherence_tensor(W_m):
    size= W_m.shape[0]
    W_norm= tf.norm(W_m,ord='euclidean',axis=1,keepdims=True)
    norm_mat=tf.divide(W_m,W_norm)
    coherence=tf.matmul(norm_mat,tf.transpose(norm_mat))-tf.diag(tf.ones([size]))
    return tf.abs(coherence)

def residual_loss(data,vectors,n_nodes):
    _loss_2=0
    if n_nodes>0:
        for i in range(n_nodes):
            _loss_2=_loss_2+get_mmse(data,vectors[i])
    return _loss_2

def get_mmse(input_, logits):
  return tf.reduce_mean( (input_ - logits) ** 2)


def get_psnr(input_, logits):
    mse = tf.reduce_mean( (input_ - logits) ** 2, axis=1 )
    return tf.reduce_mean (20 * tf.log(PIXEL_MAX / tf.sqrt(mse))/tf.log(tf.constant(10, dtype=tf.float32)))
  
    
learning = tf.placeholder('float')



fd_txt, fd_result =save_to()
  



# Input layer
x  = tf.placeholder(tf.float32, [None, image_size, image_size], name='x')
y_ = tf.placeholder(tf.int32, [None, 10],  name='y_')

phase_errors = tf.placeholder(tf.float32, [None, image_size],  name='phase_errors')

y1 = tf.cast(y_, tf.float32)

x_vector= tf.reshape(x,[-1,image_size*image_size])


keep_prob  = tf.placeholder(tf.float32)
keep_prob2  = tf.placeholder(tf.float32)


# Adler model
#x0,y,W_fc01 = model.setup_model(x, num_measurement, keep_prob)



#Our proposed model

nums=1

x0,residual_layers,residual_vectors,h_fc02= model.setup_model_autofocus(x, keep_prob, keep_prob2, phase_errors,nums)

#c_est=tf.reduce_max(coherence_tensor(W_fc01))
h_fc02_vector= tf.reshape(h_fc02,[-1,image_size*image_size])

loss_tv=tf.reduce_mean(tf.image.total_variation(residual_layers[nums-1]))

_loss_2= residual_loss(x_vector,residual_vectors,nums)+ 1e-6*loss_tv#+ get_mmse(x_vector,h_fc02_vector)
_psnr = get_psnr(x_vector,residual_vectors[nums-1])


# Evaluation functions

optimizer = tf.train.AdamOptimizer(learning).minimize(_loss_2)




#optimizer_2 = tf.train.AdamOptimizer(learning).minimize(_loss,var_list=[fc])


#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.allow_growth = True
saver = tf.train.Saver()
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True


sess = tf.Session(config=config)
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


with tf.Session() as sess:
  
  max_test_acc=0
  squ_test_acc=0
  sigma_test_acc=0
  
  max_test_acc_2=0
  squ_test_acc_2=0
  sigma_test_acc_2=0
  
  
  
  coh_avg=0
  tr_acc_avg=0
  
  for ii in range(1):
      print('Starting new dataset', ii)
      sess.run(tf.initialize_all_variables())
      max_epoch=5000
      acc_max=0
      ind_max=0
      checking=False
      
      coherence_max=0
      learning_rate=learning_rate_init
      
      tst_acc=0
      tr_acc=0
      
      
      
      for epoch in range(max_epoch):
            train_size= np.shape(trX)[0]
            num_tr_batch= np.int(np.floor(train_size/batch_size))
            
            val_size= np.shape(valX)[0]
            num_val_batch= np.int(np.floor(val_size/batch_size)+1)
            
            kk = np.arange(train_size)
            np.random.shuffle(kk)
            loss=0
            
                
            train_acc=0
            for step in range(num_tr_batch):
                start = step * batch_size
                end = start + batch_size
                noise_data=np.random.normal(0.0,0.1,(batch_size,image_size)).astype(np.float32)
                #noise_data=np.ones((batch_size,image_size,image_size)).astype(np.float32)
                #noise_data=np.random.gamma(1.0,1.0,(batch_size,image_size,image_size)).astype(np.float32)
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c, ltv = sess.run([optimizer,_loss_2,loss_tv], feed_dict={x: trX[kk[start:end]], y_: trY[kk[start:end]], keep_prob: keep_prob_value_1, keep_prob2: keep_prob_value_2, learning:learning_rate, phase_errors:noise_data})
                #w,_, c,acc, f0 = sess.run([W_fc01,optimizer_2,_loss,accuracy,fc], feed_dict={x: trX[kk[start:end]], y_: trY[kk[start:end]], keep_prob: keep_prob_value_1, keep_prob2: keep_prob_value_2, learning:1e-1})
                #acc = sess.run([accuracy], feed_dict={x: valX[start:end], y_: valY[start:end], keep_prob: 1.0,keep_prob2: 1.0})
                loss=loss+c
                
            print(ltv)
            
            val_acc = 0
            for trial in range(5):
                for step in range(num_val_batch):
                    start = step * batch_size
                    end_p = start + batch_size
                    
                    if step < num_val_batch-1:
                        test_X=valX[start:end_p]
                        test_Y=valY[start:end_p]
                        noise_data=np.random.normal(0.0,0.1,(batch_size,image_size)).astype(np.float32)
                        #noise_data=np.ones((batch_size,image_size,image_size)).astype(np.float32)
                        #noise_data=np.random.gamma(1.0,1.0,(batch_size,image_size,image_size)).astype(np.float32)
                    else:
                        test_X=valX[start:]
                        test_Y=valY[start:]
                        end_size= np.shape(test_X)[0]
                        noise_data=np.random.normal(0.0,0.1,(end_size,image_size)).astype(np.float32)
                        #noise_data=np.ones((end_size,image_size,image_size)).astype(np.float32)
                        #noise_data=np.random.gamma(1.0,1.0,(end_size,image_size,image_size)).astype(np.float32)
                    
                    #noise_data=np.zeros((2425,image_size)).astype(np.float32)
                    acc = sess.run([_psnr], feed_dict={x: test_X, y_: test_Y, keep_prob: 1.0, keep_prob2: 1.0, phase_errors:noise_data})
                        #acc=0
                    
                    val_acc += float(acc[0])
                
                
            val_acc =val_acc/(5.*num_val_batch)
            print(epoch, train_acc, loss, val_acc, ind_max, acc_max, tst_acc,tr_acc, coherence_max)
            fd_txt.write(str(epoch)+','+str(train_acc)+','+str(loss)+','+ str(val_acc)+','+str( ind_max)+','+str(acc_max)+','+str( tst_acc)+','+str(tr_acc)+'\n')
            fd_txt.flush()
            fd_result.write(str(epoch)+','+str(train_acc)+','+str(loss)+','+ str(val_acc)+','+str( ind_max)+','+str(acc_max)+'\n')
            fd_result.flush()
            if epoch ==0 or acc_max<=val_acc:
                acc_max=val_acc
                ind_max=epoch
                #coherence_max=mutual_coherence
                    

            learning_rate=learning_rate*learning_decay
            
            if epoch - ind_max>1000:
                break
      #max_test_acc=max_test_acc+acc_max
      
      









