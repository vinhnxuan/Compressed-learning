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

import cifar10_end_to_end_cl_model as model
import cifar10_modified_end_to_end_cl_model as modified_model

import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"


beta=100
keep_prob_value=0.8
learning_rate0=2e-4
learning_decay=0.996
num_measurement=8*16;
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

x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE*_IMAGE_CHANNELS], name='Input')
y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')

keep_prob  = tf.placeholder(tf.float32)

train_x, train_y, train_l = input_set.get_data_set()

test_x, test_y, test_l = input_set.get_data_set("test")


#x0, output, global_step, y_pred_cls, m_coherence = model.setup_model(x,num_measurement)
x0, output, global_step, y_pred_cls, m_coherence = modified_model.setup_model(x,num_measurement,keep_prob)

loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

loss = loss_2+ beta*m_coherence

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




def predict_test():
    '''
        Make prediction for all images in test_x
    '''
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys,keep_prob:1.0})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

    return acc

def train(num_iterations,learning_rate0):
    '''
        Train CNN
    
    '''
   
    fd_txt, fd_result =save_to()
    epoch_size=int(len(train_x)/_BATCH_SIZE)
    
    
    
    
    iterations=epoch_size*2000
    
    max_test_acc=0
    squ_test_acc=0
    sigma_test_acc=0
    iter_avg=0
    coh_avg=0
    tr_acc_avg=0
    
    for j in range(5):
        
        acc_max=0
        learning_rate=learning_rate0
        iter_max=0
        coherence_max=0
      
        tr_acc=0
        
        
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            randidx = np.random.randint(len(train_x), size=_BATCH_SIZE)
            batch_xs = train_x[randidx]
            batch_ys = train_y[randidx]
            
            start_time = time()
            i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys,learning:learning_rate,keep_prob:keep_prob_value})
            
           
            duration = time() - start_time
    
            if (i_global % epoch_size == 0) or (i == num_iterations - 1):
                mutual_coherence,_loss, batch_acc = sess.run([m_coherence,loss, accuracy], feed_dict={x: batch_xs, y: batch_ys,keep_prob:1.0})
                msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch, Mutual_coherence {5:.2f} )"
                print(msg.format(i_global, batch_acc, _loss, _BATCH_SIZE / duration, duration,coherence_max))
                fd_txt.write(str(i_global)+','+str(batch_acc)+','+str(_loss)+','+str(acc_max)+'\n')
                fd_txt.flush()
                
    
            if (i_global % epoch_size == 0) or (i == num_iterations - 1):
                acc = predict_test()
                learning_rate=learning_rate*learning_decay
    
                if acc_max==0 or acc_max<=acc:
                    acc_max=acc
                    saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
                    print("Saved checkpoint.")
                    iter_max=i_global
                    coherence_max=mutual_coherence
                    tr_acc=batch_acc
                    
                if i_global-iter_max>epoch_size*300:
                    break
                    
                print('Max test accuracy: '+ str(acc_max))
        max_test_acc=max_test_acc+acc_max
        squ_test_acc=squ_test_acc+acc_max*acc_max
        avg_test_acc=max_test_acc/(j+1)
        squ_avg_test_acc=(squ_test_acc)/(j+1)
        sigma_test_acc=np.sqrt(np.abs(avg_test_acc*avg_test_acc-squ_avg_test_acc))
        iter_avg=iter_avg+iter_max  
        coh_avg=coh_avg+coherence_max
        tr_acc_avg=tr_acc_avg+tr_acc
        coh=coh_avg/(j+1)
        tr_avg=tr_acc_avg/(j+1)
          
        print('Validation accuracy',avg_test_acc,sigma_test_acc, iter_avg/(j+1))
        fd_txt.write('Final result'+ str(avg_test_acc)+str(sigma_test_acc) + str(coh) + str(tr_avg) +'\n')
        fd_txt.flush()
    fd_txt.close()
    fd_result.close()
    return learning_rate0
        
    
if _ITERATION != 0:
    learning_rate=train(_ITERATION,learning_rate0)
    
#tf.get_variable_scope().reuse_variables()
sess.close()