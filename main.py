# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:18:38 2021

@author: lihen
"""

import time
start_time = time.time()
import numpy as np
import random
import tensorflow as tf
# Python 3.5
import os
import pandas as pd
import sys
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
result = pd.read_pickle(r'C:/Users/lihen/projects/tf-gpu-MNIST/naca4_clcd_turb_st_3para.pkl', compression=None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import pickle
os.environ['PYTHONHASHSEED']=str(1615400000)
tf.random.set_seed(1615400000)
np.random.seed(1615400000)
random.seed(1615400000)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    
#load data
inp_reno=[]
inp_aoa=[]
inp_para=[]

out_cm=[]
out_cd=[]
out_cl=[]

out_cm.extend(result[0])   
out_cd.extend(result[1])
out_cl.extend(result[2])

inp_reno.extend(result[3])
inp_aoa.extend(result[4])
inp_para.extend(result[5])

out_cm=np.asarray(out_cm)/0.188171
out_cd=np.asarray(out_cd)/0.2466741
out_cl=np.asarray(out_cl)/1.44906


out_cd=np.asarray(out_cd)
out_cl=np.asarray(out_cl)

inp_reno=np.asarray(inp_reno)
inp_aoa=np.asarray(inp_aoa)
inp_para=np.asarray(inp_para)/np.array([6,6,30])


# ---------ML PART:-----------#
#shuffle data
N= len(out_cm)
print(N)
I = np.arange(N)
np.random.shuffle(I)
n=N


#normalize
inp_reno=inp_reno/100000.
inp_aoa=inp_aoa/14.0

my_inp=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
my_out=np.concatenate((out_cd[:,None],out_cl[:,None]),axis=1)

## Training sets
xtr0= my_inp[I][:n]
ttr1 = my_out[I][:n]

# Multilayer Perceptron
# create model
xx= 90
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(5,)))
model.add(tf.keras.layers.Dense(xx, kernel_initializer='random_normal', activation=tf.keras.activations.swish))
model.add(tf.keras.layers.Dense(xx, activation=tf.keras.activations.swish))
model.add(tf.keras.layers.Dense(xx, activation=tf.keras.activations.swish))
model.add(tf.keras.layers.Dense(xx, activation=tf.keras.activations.swish))
model.add(tf.keras.layers.Dense(xx, activation=tf.keras.activations.swish))
model.add(tf.keras.layers.Dense(2, activation=None))

#callbacks
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=100, min_lr=1.0e-8)

#prevents overtraining when the ANN has converged at a local minimum
e_stop = EarlyStopping(monitor='loss', min_delta=1.0e-8, patience=200, verbose=0, mode='auto')

filepath="./model_optimized_notreduced/model_sf_{epoch:02d}_{loss:.8f}_{val_loss:.8f}.hdf5"

chkpt= ModelCheckpoint(filepath, monitor='val_loss', verbose=0,\
                                save_best_only=False, save_weights_only=False, mode='auto', period=100)

# Compile model
opt = tf.keras.optimizers.Nadam(learning_rate=0.001)

model.compile(optimizer= opt, loss=tf.keras.losses.MeanSquaredError())

hist = model.fit([xtr0], [ttr1], validation_split=0.1, batch_size=32, callbacks=[e_stop,chkpt],verbose=1,shuffle=False, epochs=10000)

#save model
model.save('./model_optimized_notreduced/final_sf.hdf5') 

print("\n") 
print("loss = %f to %f"%(np.asarray(hist.history["loss"][:1]),np.asarray(hist.history["loss"][-1:])))
print("\n")
print("val_loss = %f to %f"%(np.asarray(hist.history["val_loss"][:1]),np.asarray(hist.history["val_loss"][-1:])))
print("\n")
print("--- %s seconds ---" % (time.time() - start_time))

data1=[hist.history]
with open('./model_optimized_notreduced/hist.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)
model.summary()
#%%
model.summary()













