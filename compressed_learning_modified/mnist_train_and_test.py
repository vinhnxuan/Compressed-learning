# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:36:11 2018

@author: vinh
"""

from __future__ import print_function

import numpy as np

import tensorflow as tf

import read_dataset as input_set

import mnist_end_to_end_cl_model as model
import mnist_modified_end_to_end_cl_model as modified_model

import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

num_measurement=8;
batch_size=50
keep_prob_value_1=0.5
keep_prob_value_2=0.5
learning_rate_init=5e-4
learning_decay=0.996

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
    
learning = tf.placeholder('float')



fd_txt, fd_result =save_to()
  


trX, trY, num_tr_batch, valX, valY, num_val_batch = input_set.load_mnist(batch_size, is_training=True)
trY = trY[:num_tr_batch * batch_size].reshape((-1, 1))
#valY = valY[:num_val_batch * batch_size].reshape((-1, 1))

valX, valY, num_val_batch = input_set.load_mnist(batch_size, is_training=False)
valY = valY[:num_val_batch * batch_size].reshape((-1, 1))


# Input layer
x  = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.int32, [None, 1],  name='y_')
y1 = tf.one_hot(y_, depth=10, axis=1, dtype=tf.float32)
y1 = tf.reshape(y1,(-1, 10))

keep_prob  = tf.placeholder(tf.float32)
keep_prob2  = tf.placeholder(tf.float32)


# Adler model
#x0,y,W_fc01 = model.setup_model(x, num_measurement, keep_prob)


#Our proposed model
x0,y,W_fc01 = modified_model.setup_model(x, num_measurement, keep_prob, keep_prob2)

c_est=tf.reduce_max(coherence_tensor(W_fc01))

# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y1 * tf.log(y+1e-10), reduction_indices=[1]))

_loss=cross_entropy+ beta*c_est

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y1, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Training algorithm
optimizer = tf.train.AdamOptimizer(learning).minimize(_loss)


#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.allow_growth = True



with tf.Session() as sess:
  
  max_test_acc=0
  squ_test_acc=0
  sigma_test_acc=0
  
  max_test_acc_2=0
  squ_test_acc_2=0
  sigma_test_acc_2=0
  
  
  
  coh_avg=0
  tr_acc_avg=0
  
  for ii in range(5):
      print('Starting new dataset', ii)
      sess.run(tf.initialize_all_variables())
      max_epoch=2000
      acc_max=0
      ind_max=0
      checking=False
      
      coherence_max=0
      learning_rate=learning_rate_init
      
      tst_acc=0
      tr_acc=0
      
      
      
      for epoch in range(max_epoch):
            kk = np.arange(60000)
            np.random.shuffle(kk)
            loss=0
            
                
            train_acc=0
            for step in range(num_tr_batch):
                start = step * batch_size
                end = start + batch_size
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                w,_, c,acc = sess.run([W_fc01,optimizer,_loss,accuracy], feed_dict={x: trX[kk[start:end]], y_: trY[kk[start:end]], keep_prob: keep_prob_value_1, keep_prob2: keep_prob_value_2, learning:learning_rate})
                #acc = sess.run([accuracy], feed_dict={x: valX[start:end], y_: valY[start:end], keep_prob: 1.0,keep_prob2: 1.0})
                loss=loss+c
                train_acc=train_acc+acc
            train_acc=train_acc/num_tr_batch
            val_acc = 0
            
            mutual_coherence=compute_mcoherence(w)
            
            
            #for i in range(num_val_batch):
            #    start = i * batch_size
            #    end = start + batch_size
            acc = sess.run(accuracy, feed_dict={x: valX, y_: valY, keep_prob: 1.0, keep_prob2: 1.0})
                #acc=0
            val_acc += acc
            print(epoch, train_acc, loss, val_acc, ind_max, acc_max, tst_acc,tr_acc, coherence_max)
            fd_txt.write(str(epoch)+','+str(train_acc)+','+str(loss)+','+ str(val_acc)+','+str( ind_max)+','+str(acc_max)+','+str( tst_acc)+','+str(tr_acc)+'\n')
            fd_txt.flush()
            fd_result.write(str(epoch)+','+str(train_acc)+','+str(loss)+','+ str(val_acc)+','+str( ind_max)+','+str(acc_max)+'\n')
            fd_result.flush()
            if epoch ==0 or acc_max<=val_acc:
                acc_max=val_acc
                ind_max=epoch
                coherence_max=mutual_coherence
                tr_acc=train_acc
            
            learning_rate=learning_rate*learning_decay
            
            if epoch - ind_max>300:
                break
      #max_test_acc=max_test_acc+acc_max
      
      
      max_test_acc=max_test_acc+acc_max
      squ_test_acc=squ_test_acc+acc_max*acc_max
      avg_test_acc=max_test_acc/(ii+1)
      squ_avg_test_acc=(squ_test_acc)/(ii+1)
      sigma_test_acc=np.sqrt(np.abs(avg_test_acc*avg_test_acc-squ_avg_test_acc))
      coh_avg=coh_avg+coherence_max
      tr_acc_avg=tr_acc_avg+tr_acc
      coh=coh_avg/(ii+1)
      tr_avg=tr_acc_avg/(ii+1)
      
      print('Validation accuracy',avg_test_acc,sigma_test_acc,coh)
      fd_txt.write('Final result'+ str(avg_test_acc)+str(sigma_test_acc) + str(coh) + str(tr_avg) +'\n')
      fd_txt.flush()
  fd_txt.close()
  fd_result.close()









