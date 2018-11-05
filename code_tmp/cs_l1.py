# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:36:11 2018

@author: vinh
"""

from __future__ import print_function

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

import mstar_read_dataset as input_set

import mstar_model as model

import os

import scipy.misc as im



from scipy.linalg import dft

_SAVE_PATH = "./tensorboard/MSTAR/"


os.environ["CUDA_VISIBLE_DEVICES"]="3"

num_measurement=8;
batch_size=25
keep_prob_value_1=0.8
keep_prob_value_2=0.02
learning_rate_init=0.2
learning_decay=0.9998

image_size=256

trX, trY = input_set.get_google_data('train',width=image_size, height=image_size, crop_size=image_size)

valX, valY = input_set.get_mstar_data('test',width=image_size, height=image_size, crop_size=image_size)

PIXEL_MAX1= np.amax(trX)
PIXEL_MAX2= np.amax(valX)



def read_image_test():
    X = []
    tmp_dir = "sar_image/"
    img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpg")]
    for j in range(len(img_idx)):
        img = im.imread((tmp_dir + img_idx[j]))
        img = np.mean(img,axis=2)
        width_org= np.shape(img)[0]
        img = img[0:,0:width_org]
        X.append(img)
        
    return np.asarray(X)


image_arr= read_image_test()

#PIXEL_MAX=np.maximum(PIXEL_MAX1, PIXEL_MAX2)
PIXEL_MAX=np.amax(image_arr)

print(PIXEL_MAX)

# regularization factor
beta=100

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def output_measurement_cs(dataX, phase_errors,speck_errors,batch_cs_matrix):
    n= image_size
    
    dft_m = (dft(n)/np.sqrt(n))
    
    x_speckled=np.multiply(dataX, speck_errors)
    
    dft_x=np.matmul(x_speckled,dft_m)
    
    aa= np.multiply(np.pi,phase_errors)
    
    phase_error_amp= np.exp(-2j*aa)
    
    multiply = [1,n]
###
    phase_error_matrix = np.transpose(np.reshape(np.tile(phase_error_amp, multiply), [n, n]))
##
     
    
    dft_y=np.multiply(dft_x,phase_error_matrix)
    
    
    dft_y_cs=np.multiply(dft_y,(batch_cs_matrix))

    return dft_y_cs

def debuchie4_matrix(N):
    
    h1=(1+np.sqrt(3))/(4*np.sqrt(2))
    h2=(3+np.sqrt(3))/(4*np.sqrt(2))
    h3=(3-np.sqrt(3))/(4*np.sqrt(2))
    h4=(1-np.sqrt(3))/(4*np.sqrt(2))
    
    g1=h4
    g2=-h3
    g3=h2
    g4=-h1
    
    d = [h1,h2,h3,h4]
    d2 = [g1,g2,g3,g4]
    
    e= []
    
    L=N-4
    iters=int(N/2)
    
    for i in range(iters):
        if L-2*i>=0:
            e.append( np.pad(d,(i*2,L-2*i),'constant', constant_values=(0)))
            e.append( np.pad(d2,(i*2,L-2*i),'constant', constant_values=(0)))
        else:
            e1 = np.pad(d,(i*2,0),'constant', constant_values=(0))
            e.append(e1[:N])
            e2 = np.pad(d2,(i*2,0),'constant', constant_values=(0))
            e.append(e2[:N])
        
        
    f = np.concatenate(e)
    
    f = np.reshape(f,[N,N])
    return f

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
cs_matrix = np.load("sar_cs_mat.dat")

#cs_matrix = np.ones([image_size,image_size])
  
noise_data_autofocus=np.random.normal(0.0,0.05,(image_size,1)).astype(np.float32)
#noise_data_speckling=np.random.gamma(100.0,1.0,(batch_size,image_size,image_size)).astype(np.float32)/100.
noise_data_speckling=np.ones([image_size,image_size])
               
data_num = np.shape(valX)[0]
y_m={}
 
for i in range (100):
    y_m[i] = output_measurement_cs (np.asarray(valX[i]),noise_data_autofocus,noise_data_speckling,cs_matrix) 


# Input layer
x  = tf.placeholder(tf.float32, [image_size, image_size], name='x')
y  = tf.placeholder(tf.complex128, [image_size, image_size], name='y')

est_img = weight_variable([image_size, image_size]) 

phi = tf.constant(np.transpose(debuchie4_matrix(256)))




est_img = tf.cast(est_img, dtype=tf.complex128)

dft_m = tf.constant((dft(image_size)/np.sqrt(image_size)))

dft_img = (tf.matmul(est_img,dft_m))

cs_matrix_tf= tf.constant(cs_matrix)

dft_img_cs=tf.multiply(dft_img,cs_matrix)

vest_img = tf.reshape(tf.cast(est_img, dtype=tf.float32),[1,image_size,image_size,1])

loss_tv=tf.reduce_mean(tf.image.total_variation(vest_img))

loss_tv= tf.cast(loss_tv, dtype=tf.float64)

_loss_2 = tf.reduce_mean( tf.square(tf.abs(y - dft_img_cs)))

est_img = tf.cast(est_img, dtype=tf.float64)

_loss_3 = tf.reduce_mean(tf.abs(tf.matmul(phi,est_img)))

#loss_tv = tf.reduce_mean(-tf.multiply(est_img,tf.log(tf.abs(est_img)+1)))

_loss = _loss_2 + 2e-4*loss_tv #+ 0.1*_loss_3


loss_img = tf.reduce_mean((x-vest_img)**2) 

psnr = tf.reduce_mean (20 * tf.log(PIXEL_MAX / tf.sqrt(loss_img))/tf.log(tf.constant(10, dtype=tf.float32)))

optimizer = tf.train.AdamOptimizer(learning,beta1=0.1,beta2=0.1).minimize(_loss)




#optimizer_2 = tf.train.AdamOptimizer(learning).minimize(_loss,var_list=[fc])


#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.allow_growth = True
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True


sess = tf.Session(config=config)


noise_var=0.5



image_cs={}
for i in range (np.shape(image_arr)[0]):
    image_cs[i] = output_measurement_cs (image_arr[i],noise_data_autofocus,noise_data_speckling,cs_matrix)



with tf.Session() as sess:
  
  max_test_acc=0
  squ_test_acc=0
  sigma_test_acc=0
  
  max_test_acc_2=0
  squ_test_acc_2=0
  sigma_test_acc_2=0
  
  
  
  coh_avg=0
  tr_acc_avg=0
  
  psnr_avg=0
  
  for ii in range(1):
      print('Starting new dataset', ii)
      sess.run(tf.initialize_all_variables())
      max_epoch=50000
      acc_max=0
      ind_max=0
      checking=False
      
      coherence_max=0
      learning_rate=learning_rate_init
      
      tst_acc=0
      tr_acc=0
      
      
      
      for epoch in range(max_epoch):
            
            _, c, snr,img,c2,c3 = sess.run([optimizer,_loss_2, psnr,vest_img,_loss_3,loss_tv], feed_dict={x: image_arr[5], y: image_cs[5], learning:learning_rate})

            learning_rate=learning_rate*learning_decay
            
            
            
            if acc_max==0 or acc_max<snr:
                acc_max=snr
                ind_max=epoch
                
            #print("Max psnr", acc_max, " current psnr ", snr)
            
            
            if epoch - ind_max>1000:
                break
      #max_test_acc=max_test_acc+acc_max
      psnr_avg=psnr_avg+acc_max
      print(psnr_avg/(ii+1))
      print(c,c2,snr)
      img = np.reshape(img,[image_size,image_size])
      
      dft_matrix = dft(image_size)/np.sqrt(image_size)
      
      result = np.real(np.matmul(image_cs[2],np.asmatrix(dft_matrix).getH()))
      
      
      #img = np.matmul(np.transpose(debuchie4_matrix(image_size)),valX[ii])
      
      plt.imshow(image_arr[2])
      plt.gray()
      plt.show()
      
      plt.imshow(img)
      plt.gray()
      plt.show()
      
      plt.imshow(result)
      plt.gray()
      plt.show()
      
  psnr_avg=psnr_avg/100

    






