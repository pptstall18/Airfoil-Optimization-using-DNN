# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:00:41 2021

@author: lihen
"""
import tensorflow as tf
import numpy as np



model=tf.keras.models.load_model(r'C:/Users/lihen/projects/tf-gpu-MNIST/model3x50/relu/final_sf.hdf5')


parameter = [[5.88429687, 7.80370447, 10.18074411]]
for i in parameter:  
    mypara=np.array(i)/np.array([6,6,30])
    reno=np.asarray([30000])/100000.
    aoa=np.asarray([10])/14 
    my_inp=np.concatenate((reno,aoa,mypara),axis=0)
    my_inp=np.reshape(my_inp,(1,5))
    print(model.predict([my_inp])*[0.2466741,1.44906])