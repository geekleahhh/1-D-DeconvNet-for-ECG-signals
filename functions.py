#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:46:02 2019

@author: yaoweili
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import tensorflow as tf


'''
This file contains functions as follow:
| functions                 | Usage                                                 |
| multicolored_lines        | plot deconvolution values as color on original signal |
| colorline                 | plot deconvolution values as color on original signal |
| make_segments             | plot deconvolution values as color on original signal |
| zscore                    | preprocessing functions                               |
| unpooling                 | unpooling function, keep the pooling position         |
|                             ... and others as zero values                         |
'''

#--------------------------plotting function--------------------------------

def multicolored_lines(x,y,z,layer,img_id,model_id):

#    fig, ax = plt.subplots(figsize=[20,2])
    fig=plt.figure(figsize=[60,2])
    norm=plt.Normalize(-z.max(), z.max())
    lc = colorline(x, y,z[1800:3500], cmap='coolwarm',norm=norm)
#    lc = colorline(x,y,z, cmap='coolwarm',norm=norm)

    plt.colorbar(lc)
    if layer==None:
#        plt.title('1-layer ConvNet-model-{}'.format(model_id))
        plt.title('Example of Noise beat')
    else:
        plt.title('layer'+str(layer))
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
#    plt.show()
    return fig

def colorline(x, y, z, cmap, norm,
        linewidth=3, alpha=1.0):
    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


#--------------------------preprocessing function--------------------------------
def zscore(data):
    data_mean=np.mean(data)
    data_std=np.std(data, axis=0)
    if data_std!=0:
        data=(data-data_mean)/data_std
    else:
        data=data-data_mean
    return data

#--------------------------unpooling function--------------------------------
def unpooling(relu,index,pool):
    b, h, w, c = relu.shape.as_list()
    shape=tf.constant([b * h * w * c], dtype=tf.int64)

    try:
        b2, h2, w2=pool.shape.as_list()
        pool=tf.reshape(pool,[b2,h2,w2,1])
    except:
        print('no need to reshape')   
    unpool_flattened = tf.scatter_nd(tf.reshape(index[:,:,:,:], [-1,1]), tf.reshape(pool[:,:,:,:], [-1]), shape)
    unpool=tf.reshape(unpool_flattened,[1,h,-1])
    return unpool
