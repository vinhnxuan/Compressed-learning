#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:01:43 2018

@author: gw438
"""

import os 
import numpy as np
import scipy.misc as im

from scipy.io import loadmat 

from scipy.ndimage.interpolation import rotate

import matplotlib.pyplot as plt

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot




def get_google_data(stage, width=128, height=128, crop_size=128, aug=False):
    data_dir = "data_set/"
    print("------ " + stage + " ------")
    sub_dir = ["maps", "cityscapes"]
    X = []
    y = []

    for i in range(len(sub_dir)):
        data_type = "train/" if stage == "train" else "val/" if stage == "test" else None
        tmp_dir = data_dir + sub_dir[i] + "/" + data_type
        img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpg")]
        
        for j in range(len(img_idx)):
            if stage == "train" : 
                if j ==200:
                    break;
                for k in range(1):
                    if k ==0:
                        img = im.imread((tmp_dir + img_idx[j]))
                        img = np.mean(img,axis=2)
                        width_org= np.shape(img)[0]
                        img = img[0:,0:width_org]
                        img = resize(img, [height, width])
                        
                    else:
                        img = im.imread((tmp_dir + img_idx[j]))
                        img = np.mean(img,axis=2)
                        width_org= np.shape(img)[0]
                        img = img[0:,0:width_org]
                        img = resize(img, [height, width])
                        
                        r_width = np.random.randint(7, 10)
                        r_height = np.random.randint(7, 10)
                        
                        crop_size=[int(height*r_height/(r_height+1)), int(width*r_width/(r_width+1))]
                        img = random_crop(img, crop_size)
                        #img = horizontal_flip(img)
                        img = random_rotation(img, angle_range=(0, 5))
                        img = resize(img, [height, width])
                    
    #                img = img[(height - crop_size) // 2 : height - (height - crop_size) // 2, \
    #                      (width - crop_size) // 2: width - (width - crop_size) // 2]
                    # img = img[16:112, 16:112]   # crop
#                    img = img/128.
#                    img = img -1
                    X.append(img)
                    y += [i]
                
            else:
                
                img = im.imread((tmp_dir + img_idx[j]))
                img = np.mean(img,axis=2)
                width_org= np.shape(img)[0]
                img = img[0:,0:width_org]
                img = resize(img, [height, width])
                
#                img = img/128.
#                img = img -1
                #img = img[16:112, 16:112]   # crop
                X.append(img)
                y += [i]
                
                if j ==100:
                        break;

    return np.asarray(X), dense_to_one_hot(np.asarray(y))


def get_google_raw_data(stage, width=128, height=128, crop_size=128, aug=False):
    data_dir = "data_set/"
    print("------ " + stage + " ------")
    sub_dir = ["maps"]
    X = []
    y = []
    Z=[]

    for i in range(len(sub_dir)):
        data_type = "train/" if stage == "train" else "val/" if stage == "test" else None
        tmp_dir = data_dir + sub_dir[i] + "/" + data_type
        img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".mat")]
        
        for j in range(len(img_idx)):
            
            filename=os.path.splitext(img_idx[j])[0]
            
            if stage == "train" : 
                if j ==200:
                    break;
                for k in range(1):
                    if k ==0:
                        img = loadmat(tmp_dir + img_idx[j])['data']
                        na_img = im.imread(tmp_dir +(filename + '.jpg'))
                        na_img = np.mean(na_img,axis=2)
                        width_org= np.shape(na_img)[0]
                        na_img = na_img[0:,0:width_org]
                        na_img = resize(na_img, [height, width])
#                        img = np.mean(img,axis=2)
#                        width_org= np.shape(img)[0]
#                        img = img[0:,0:width_org]
#                        img = resize(img, [height, width])
                        
                    else:
                        img = loadmat(tmp_dir + img_idx[j])['data']
                        na_img = im.imread(tmp_dir +(filename + '.jpg'))
                        na_img = np.mean(na_img,axis=2)
                        width_org= np.shape(na_img)[0]
                        na_img = na_img[0:,0:width_org]
                        na_img = resize(na_img, [height, width])
#                        img = np.mean(img,axis=2)
#                        width_org= np.shape(img)[0]
#                        img = img[0:,0:width_org]
#                        img = resize(img, [height, width])
#                        
#                        r_width = np.random.randint(7, 10)
#                        r_height = np.random.randint(7, 10)
#                        
#                        crop_size=[int(height*r_height/(r_height+1)), int(width*r_width/(r_width+1))]
#                        img = random_crop(img, crop_size)
#                        #img = horizontal_flip(img)
#                        img = random_rotation(img, angle_range=(0, 5))
#                        img = resize(img, [height, width])
                    
    #                img = img[(height - crop_size) // 2 : height - (height - crop_size) // 2, \
    #                      (width - crop_size) // 2: width - (width - crop_size) // 2]
                    # img = img[16:112, 16:112]   # crop
#                    img = img/128.
#                    img = img -1
                    X.append(img)
                    Z.append(na_img)
                    y += [i]
                
            else:
                
                img = loadmat(tmp_dir + img_idx[j])['data']
                na_img = im.imread(tmp_dir +(filename + '.jpg'))
                na_img = np.mean(na_img,axis=2)
                width_org= np.shape(na_img)[0]
                na_img = na_img[0:,0:width_org]
                na_img = resize(na_img, [height, width])
                
#                img = img/128.
#                img = img -1
                #img = img[16:112, 16:112]   # crop
                X.append(img)
                Z.append(na_img)
                y += [i]
                
                if j ==100:
                        break;

    return np.asarray(X), np.asarray(Z)




def get_mstar_data(stage, width=64, height=64, crop_size=64, aug=False):
    data_dir = "data_set/MSTAR-10/train/" if stage == "train" else "data_set/MSTAR-10/test/" if stage == "test" else None
    print("------ " + stage + " ------")
    sub_dir = ["2S1", "BMP2", "BRDM_2", "BTR60", "BTR70", "D7", "T62", "T72", "ZIL131", "ZSU_23_4"]
    X = []
    y = []

    for i in range(len(sub_dir)):
        tmp_dir = data_dir + sub_dir[i] + "/"
        img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpeg")]
        
        for j in range(len(img_idx)):
            if stage == "train" : 
                for k in range(10):
                    if k ==0:
                        img = resize(im.imread((tmp_dir + img_idx[j])), [height, width])
                    else:
                        img = resize(im.imread((tmp_dir + img_idx[j])), [height, width])
                        r_width = np.random.randint(7, 10)
                        r_height = np.random.randint(7, 10)
                        
                        crop_size=[int(height*r_height/(r_height+1)), int(width*r_width/(r_width+1))]
                        img = random_crop(img, crop_size)
                        #img = horizontal_flip(img)
                        img = random_rotation(img, angle_range=(0, 5))
                        img = resize(img, [height, width])
                    
    #                img = img[(height - crop_size) // 2 : height - (height - crop_size) // 2, \
    #                      (width - crop_size) // 2: width - (width - crop_size) // 2]
                    # img = img[16:112, 16:112]   # crop
#                    img = img/128.
#                    img = img -1
                    X.append(img)
                    y += [i]
            else:
                img = resize(im.imread((tmp_dir + img_idx[j])), [height, width])
#                img = img/128.
#                img = img -1
                #img = img[16:112, 16:112]   # crop
                X.append(img)
                y += [i]
                if j ==10:
                        break;

    return np.asarray(X), dense_to_one_hot(np.asarray(y))

def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


def subtract(image):
    image = image / 255
    return image


def resize(image, size):
    image = im.imresize(image, size)
    return image


def center_crop(image, crop_size):
    h, w, _ = image.shape
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def random_crop(image, crop_size):
    h, w = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right]
    return image


def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1]
    return image


def vertical_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
    return image


def scale_augmentation(image, scale_range, crop_size):
    scale_size = np.random.randint(*scale_range)
    image = im.imresize(image, (scale_size, scale_size))
    image = random_crop(image, crop_size)
    return image


def random_rotation(image, angle_range=(0, 180)):
    h, w = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = resize(image, (h, w))
    return image


def cutout(image_origin, mask_size, mask_value='mean'):
    image = np.copy(image_origin)
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    image[top:bottom, left:right, :].fill(mask_value)
    return image


def random_erasing(image_origin, p=0.5, s=(0.02, 0.4), r=(0.3, 3), mask_value='random'):
    image = np.copy(image_origin)
    if np.random.rand() > p:
        return image
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    image[top:bottom, left:right, :].fill(mask_value)
    return image


x,y = get_google_raw_data("train")

plt.imshow(np.abs(x[2]))
plt.gray()
plt.show()

##
#from scipy.linalg import dft   
#
#n=256 
##
#X,Y = get_mstar_data("test",width=n, height=n, crop_size=n)
##
##
#phase_errors=np.random.normal(0.0,0.5,(n,1)).astype(np.float32)
##
#dft_matrix= dft(n)/np.sqrt(n)
#
#dft_matrix_h = np.asmatrix(dft_matrix).getH()
##    
#dft_x=np.matmul(X[0],dft_matrix)
#
#aa=np.multiply(np.pi,phase_errors)
###
#phase_error_amp= np.exp(2j*aa)
###
#multiply = [1,n]
###
#phase_error_matrix = np.transpose(np.reshape(np.tile(phase_error_amp, multiply), [n, n]))
##
##
##
### 
###phase_error_matrix = tf.cast(phase_error_matrix, dtype=tf.complex64)
###
#dft_y=np.multiply(dft_x,phase_error_matrix)
#
#a = np.matmul(dft_y,np.asmatrix(dft_matrix).getH())
#
#
#result = np.real(a)
#
#
#plt.imshow(X[0])
#plt.gray()
#plt.show()
#
#plt.imshow(result)
#plt.gray()
#plt.show()
#
#plt.imshow(np.real(dft_y))
#plt.gray()
#plt.show()
