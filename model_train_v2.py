# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:36:49 2019

@author: yaoweili
"""

# -*- coding: utf-8 -*-


import numpy as np
import scipy.io as sci
import glob
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM, Flatten, Conv1D, Dropout,BatchNormalization,GRU,MaxPooling1D,LeakyReLU
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import minmax_scale 
import numpy as np
from scipy.fftpack import fft, ifft
import math
import scipy.signal as signal
import scipy.io as sio
import os

ACC_path='./model/lossandacc/'
if not os.path.exists(ACC_path):
    os.makedirs(ACC_path)

i=1

def zscore(data):
    data_mean=np.mean(data)
    data_std=np.std(data, axis=0)
    if data_std!=0:
        data=(data-data_mean)/data_std
    else:
        data=data-data_mean
    return data


def all_data(datapath):
    training_data=[]
    labels=[]
    label=[]
    train_set = glob.glob(datapath+ '*.mat')
    for image_fn in train_set:
        image_type = image_fn.split('/')[-1][:1]
        if image_type == 'A':
            label = 0
        if image_type == 'N':
            label =1
        if image_type == 'V':
            label = 2
        if image_type == 'C':
            label = 3
        labels.append(label)
        ecg=sci.loadmat(image_fn)
        data_a=ecg['ecg']
        data_a=np.reshape(data_a,(5000))
        data_a=zscore(data_a)
        data_a=minmax_scale(data_a,axis=0, feature_range=(0, 5))
        data_a=data_a.flatten()
        training_data.append(data_a)
    training_data=np.asarray(training_data)
    labels = np.asarray(labels)
    training_data=training_data.astype("float32")
    labels=labels.astype("int")
    return training_data,labels

x,y=all_data('../data/')

skf= StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
for train,test in skf.split(x,y):
    x_train=x[train]
    y_train=y[train]
    x_test=x[test]
    y_test=y[test]
    x_train_r=np.reshape(x_train,[-1,5000,1])
    x_test_r=np.reshape(x_test,[-1,5000,1])
    y_train_r=to_categorical(y_train)
    y_test_r=to_categorical(y_test)
    model = Sequential()

    model.add(Conv1D(4,35, input_shape=(5000,1), padding="same", strides=1))
    model.add(MaxPooling1D(pool_size=16, strides=None, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('softmax'))
   
    sgd = SGD(lr=0.03, nesterov=True,decay=1e-4, momentum=0.90)
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    model.summary ()
    
    nb_epoch = 200
    test_loss=[]
    test_acc=[]
    train_loss=[]
    train_acc=[]
    loss_last=10.0
    
    for e in range(nb_epoch):
        history = model.fit(x=x_train_r, y=y_train_r,batch_size=500, epochs=1,verbose=2)
        loss,acc=model.evaluate(x_test_r,y_test_r,steps=1, verbose=1)
        print("epoch_",e,"test loss",loss,"test acc",acc)
        test_loss.append(loss)
        test_acc.append(acc)
        trainloss=history.history["loss"]
        trainacc=history.history["acc"]
        train_loss.append(trainloss)
        train_acc.append(trainacc)
        filepath="./model/model"+str(i)+"/"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        model.save(filepath+"model"+str(e)+"-weights-imporvement.hdf5") 
    i=i+1
