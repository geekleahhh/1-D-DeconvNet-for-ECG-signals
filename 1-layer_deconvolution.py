#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:31:41 2019

@author: yaoweili
"""


import tensorflow as tf
import keras
import numpy as np
import scipy.io as sci
import glob
import matplotlib.pyplot as plt
import functions as ff
import os

from sklearn.preprocessing import minmax_scale 
from scipy.fftpack import fft, ifft


datapath='../data/'

model_path='./model/model1/model0-weights-imporvement.hdf5'


model_id=model_path.split('/')[-1].split('.')[-2]
part='_Rpeak'
output_path='./figure/1layer_model_{}/'.format(model_id)

if not os.path.exists(output_path):
    os.makedirs(output_path)


#========================================preprocessing==============================



def inputs(datapath):
    training_data=[]
    labels=[]
    label=[]
    train_set = glob.glob(datapath+ 'N1.mat')
    img_id=0
    for image_fn in train_set:
        image_type = image_fn.split('/')[-1][:1]
        if image_type == 'A':
            label = [0,0,1,0]
        elif image_type == 'V':
            label =[1,0,0,0]
        elif image_type == 'N':
            label = [0,1,0,0]
        elif image_type == 'C':
            label = [0,0,0,1]
        img_id=image_fn.split('/')[-1].split('.')[0]
        print(img_id)
        labels.append(label)
        ecg=sci.loadmat(image_fn)
        data_a=ecg['ecg']
        data_a=np.reshape(data_a,(5000))
        data_a=ff.zscore(data_a)
        data_a=data_a.flatten()
        training_data.append(data_a)

    training_data=np.asarray(training_data)
    labels = np.asarray(labels)
    training_data=training_data.astype("float32")
    labels=labels.astype("int")
    return training_data,labels,img_id



#================================== load data&model =================================
#load data
data,label,img_id=inputs(datapath)
data=np.reshape(data,[-1,5000,1])
print(data.shape)

inputImage=tf.placeholder(tf.float32, shape=[1, 5000, 1])
labels = tf.placeholder(tf.uint8, shape=[1,4])

#load model in endpoints
model = keras.models.load_model(model_path)
sess = keras.backend.get_session()
endpoints = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)


#convolution---------------filtering
conv1_weights=endpoints[0]
conv1_biases=endpoints[1]
conv1 = tf.nn.conv1d(inputImage,conv1_weights,stride=1,padding="SAME") #
fc1 = conv1+conv1_biases
relu1 = tf.nn.relu(fc1)
relu1=tf.reshape(relu1,[1,5000,-1,1])

#convolution---------------pooling
pool1,index1 = tf.nn.max_pool_with_argmax(relu1,ksize=[1,16,1,1],padding="SAME",strides=[1,16,1,1])
pool1=tf.reshape(pool1,[1,313,-1])




#----------------------------------------Deconvolution----------------------------------
"""
    # ============= Deconvoluton: 3 parts ==================
    #----- unpooling
    #----- unrelu
    #----- unconv
"""

#Deconvolution-------------unpooling
unpool1=ff.unpooling(relu1,index1,pool1)

#Deconvolution-------------unrelu
unrelu1=tf.nn.relu(unpool1)

#Deconvolution-------------unconv
b, h, c= inputImage.shape.as_list()
unconv1=tf.contrib.nn.conv1d_transpose(unrelu1,conv1_weights,padding="SAME",stride=1,output_shape=[b,h,c])


activations1 = unconv1.eval(feed_dict={inputImage: data, labels: label},session=sess)
sess.close()


#----------------------------------------plotting----------------------------------

#-------------------------------plot original signal
#-------------------------------plot deconvolution curve
"""
    # ===============================
    # x-axis[variable:None] : choose a rage, here is just the original range: 0-5000
    # y-axis[variable:x_or] : orginal ECG signal
    # ===============================
"""
x_or=np.reshape(data,[-1])
fig_or=plt.figure(figsize=[15,2])
plt.plot(x_or)
plt.title('original signal')
fig_or.savefig(output_path+img_id+'_or.png',quality=100,subsampling=0)

#-------------------------------plot deconvolution curve
"""
    # ===============================
    # x-axis[variable:None] : choose a rage, here is just the original range: 0-5000
    # y-axis[variable:activations1] : Deconvolution values 
    # ===============================
"""
activations1=np.reshape(activations1,[-1])
fig_deconv=plt.figure(figsize=[40,2])
plt.plot(activations1)
plt.title('Example of Noise beat'.format(model_id))
fig_deconv.savefig(output_path+img_id+'deconvnet'+'model'+model_id+part+'.png',quality=100,subsampling=0)



#-------------------------------plot decovolution values as color on original signal
"""
    # ===============================
    # x-axis[variable:x] : choose a rage, here we choose 1800-3499
    # y-axis[variable:y]: ECG values
    # color[variable:z]: Deconvolution values 
    # ===============================
"""
x = np.linspace(1800, 3499,1700)
#x=np.linspace(0,4999,5000)
y = x_or[1800:3500]
#y = x_or
z = activations1
figc=ff.multicolored_lines(x,y,z,None,img_id,model_id)
figc.savefig(output_path+img_id+'1-layer_ConvNet_'+'model'+model_id+part+'.png',quality=100,subsampling=0)


