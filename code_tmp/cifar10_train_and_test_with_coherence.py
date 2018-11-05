# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 10:49:26 2018

@author: vinh
"""

from __future__ import print_function

import numpy as np

from time import time

import tensorflow as tf

import read_dataset as input_set

import cifar10_modified_end_to_end_cl_model as modified_model

import scipy.io as sci
# generative models


import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"


beta=0.05
keep_prob_value=0.8
learning_rate0=4e-4
learning_decay=0.998
num_measurement=300;
_BATCH_SIZE = 128
_CLASS_SIZE = 10
_ITERATION = 10000

_IMAGE_SIZE = 32
_IMAGE_CHANNELS = 3
_NUM_CLASSES = 10
_NUM_CHANNELS = 3

_SAVE_PATH = "./tensorboard/cifar-10/"
learning = tf.placeholder('float')

def save_to():
    export_file_txt="tmp/train_txt_cifar10_"+str(num_measurement)+".txt"
    export_file_result = "tmp/train_txt_cifar10_"+str(num_measurement)+".csv"


    fd_txt = open(export_file_txt, 'w')
    fd_txt.write('End to end CNN\n')
    fd_result = open(export_file_result, 'w')
    fd_result.write('epoch,train_acc,loss,val_acc,val_max\n')
    return(fd_txt, fd_result)
    
def get_mmse(input_, logits):
  return tf.reduce_mean( (input_ - logits) ** 2)

def get_psnr(input_, logits):
    mse = tf.reduce_mean( (input_ - logits) ** 2, axis=1 )
    PIXEL_MAX = 1.0
    
    return tf.reduce_mean (20 * tf.log(PIXEL_MAX / tf.sqrt(mse))/tf.log(tf.constant(10, dtype=tf.float32)))
  

x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE*_IMAGE_CHANNELS], name='Input')
y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')

noise_x  = tf.placeholder(tf.float32, [None, _IMAGE_SIZE * _IMAGE_SIZE*_IMAGE_CHANNELS])


keep_prob  = tf.placeholder(tf.float32)

train_x, train_y, train_l = input_set.get_data_set()

test_x, test_y, test_l = input_set.get_data_set("test")

#x0, output, global_step, y_pred_cls, m_coherence, W_01 = model.setup_model(x,num_measurement)
x0, output, global_step, y_pred_cls, m_coherence, W_01, list_recognition = modified_model.setup_model(x+noise_x,num_measurement,keep_prob)


MM = tf.square(tf.reduce_mean((W_01)))


ones= np.ones(_IMAGE_SIZE*_IMAGE_SIZE).astype(np.float32)

a1= tf.reduce_sum(tf.multiply(tf.matmul(W_01,tf.transpose(W_01)),ones),axis=1)

aa= tf.reduce_mean(tf.square(a1-1))



loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

loss = loss_2 + (1-beta)*aa#+ beta*m_coherence #+ 0.002*MM

optimizer = tf.train.AdamOptimizer(learning).minimize(loss, global_step=global_step)

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)

merged = tf.summary.merge_all()
saver = tf.train.Saver()
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True


sess = tf.Session(config=config)
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)




def predict_test(W):
    '''
        Make prediction for all images in test_x
    '''
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    
    psnr_total=0
    count=0
    nx_array= np.zeros((10000,_IMAGE_SIZE * _IMAGE_SIZE*_IMAGE_CHANNELS)).astype(np.float32)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        nx= nx_array[i:j, :]
        predicted_class[i:j], psnr_batch = sess.run([y_pred_cls,m_coherence], feed_dict={x: batch_xs, y: batch_ys,noise_x:nx,keep_prob:keep_prob_value})
        i = j
        psnr_total=psnr_total+psnr_batch
        count=count+1
    
    
    psnr_total=psnr_total/count

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

    return acc

def train(num_iterations):
    '''
        Train CNN
    
    '''
   
    #fd_txt, fd_result =save_to()
    epoch_size=int(len(train_x)/_BATCH_SIZE)
    
    
    
    max_test_acc=0
    squ_test_acc=0
    sigma_test_acc=0
    iter_avg=0

    tr_acc_avg=0
    
    for j in range(1):
        learning_rate=learning_rate0
        acc_max=0
        #activate=False;
        iter_max=0
        psnr_max_noise=0
        psnr_max=0
      
        tr_acc=0
        
        
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            randidx = np.random.randint(len(train_y), size=len(train_y))
            total_loss=0
            total_duration=0
            avg_train=0
            
            nx_array= np.random.normal(0.0,0.05,(50000,_IMAGE_SIZE * _IMAGE_SIZE*_IMAGE_CHANNELS)).astype(np.float32)
            
            for k in range(epoch_size):
                start_pos= k*_BATCH_SIZE
                end_pos=k*_BATCH_SIZE + _BATCH_SIZE
            
                batch_xs = train_x[randidx[start_pos:end_pos]]
                batch_ys = train_y[randidx[start_pos:end_pos]]
                nx = nx_array[randidx[start_pos:end_pos]]
                
            
                start_time = time()
                _,_loss, batch_acc, w_opt, coh = sess.run([optimizer,loss, accuracy,W_01, m_coherence], feed_dict={x: batch_xs, y: batch_ys,noise_x: nx,learning:learning_rate,keep_prob:keep_prob_value})
            
            
                duration = time() - start_time
                total_duration=total_duration+duration
                total_loss=total_loss+_loss
                avg_train=avg_train+batch_acc
            avg_train=avg_train/epoch_size 
            
            acc = predict_test(w_opt)
                
            msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/epoch), max step {5:>6}, mcoh {6:.2f}"
            print(msg.format(i, avg_train, total_loss, _BATCH_SIZE / duration, total_duration, iter_max, coh))
            #fd_txt.write(str(i)+','+str(avg_train)+','+str(total_loss)+','+str(acc_max)+'\n')
            #fd_txt.flush()
                
                
            
            
            
            learning_rate=learning_rate*learning_decay

            if acc_max==0 or acc_max<=acc :
                acc_max=acc
                saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
                print("Saved checkpoint.")
                iter_max=i
                tr_acc=avg_train
                sci.savemat('W_matrix_opt_tv_coherence12_'+str(beta)+"_"+str(num_measurement)+"_"+str(j)+'.mat', {'w':w_opt})
                
            if i-iter_max>500:
                break
                    
            print('Max test accuracy: '+ str(acc_max))
              
       
        
        max_test_acc=max_test_acc+acc_max
        squ_test_acc=squ_test_acc+acc_max*acc_max
        avg_test_acc=max_test_acc/(j+1)
        squ_avg_test_acc=(squ_test_acc)/(j+1)
        sigma_test_acc=np.sqrt(np.abs(avg_test_acc*avg_test_acc-squ_avg_test_acc))
        iter_avg=iter_avg+iter_max  
        tr_acc_avg=tr_acc_avg+tr_acc
        #coh=coh_avg/(j+1)
        #tr_avg=tr_acc_avg/(j+1)
          
        print('Validation accuracy',avg_test_acc,sigma_test_acc, iter_avg/(j+1))
        #fd_txt.write('Final result'+ str(avg_test_acc)+str(sigma_test_acc) + str(coh) + str(tr_avg) +'\n')
        #fd_txt.flush()
    #fd_txt.close()
    #fd_result.close()
    return acc_max,psnr_max, psnr_max_noise



    
if _ITERATION != 0:
    learning_rate=train(_ITERATION)
    
#tf.get_variable_scope().reuse_variables()
sess.close()