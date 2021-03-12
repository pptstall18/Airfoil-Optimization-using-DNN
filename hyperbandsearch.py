# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:47:18 2021

@author: lihen
"""

import time
start_time = time.time()
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import random
import tensorflow as tf
# Python 3.5
import numpy as np
import pandas as pd
import sys
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
result = pd.read_pickle(r'C:/Users/lihen/projects/tf-gpu-MNIST/naca4_clcd_turb_st_3para.pkl', compression=None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import kerastuner as kt
import pickle
os.environ['PYTHONHASHSEED']=str(1)
tf.random.set_seed(1615400000)
np.random.seed(1615400000)
random.seed(1615400000)


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
xtr0 = my_inp[I][:n]
ttr1 = my_out[I][:n]



def model_builder(hp):
    hp_units = hp.Int('units', min_value=10, max_value=100)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(5,)))
    model.add(tf.keras.layers.Dense(units=hp_units, kernel_initializer='random_normal', activation='relu'))
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), min_value=10, max_value=100), activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation='linear'))
    
        
        
    hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 2.5e-5])
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
      loss='mean_squared_error',
      metrics=['MeanSquaredError']
      )
    return model  

e_stop = EarlyStopping(monitor='val_loss', min_delta=1.0e-8, patience=200, verbose=1, mode='auto')

tuner = kt.Hyperband(model_builder, objective='val_loss', max_epochs=100, factor=3, directory='my_dir',project_name='intro_to_kt')
tuner.search_space_summary()
tuner.search([xtr0], [ttr1], epochs=100, validation_split=0.1, callbacks=[e_stop])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


model = tuner.hypermodel.build(best_hps)
tuner.results_summary()
history= model.fit([xtr0], [ttr1], epochs=100, validation_split=0.1)

#%%
model.summary()


#%%
hypermodel = tuner.hypermodel.build(best_hps)
hist=model.fit([xtr0], [ttr1], validation_split=0.1,
                 epochs=10000, batch_size=64,callbacks=[e_stop],verbose=1,shuffle=False)
model.save('./modelx/relu/final_sf.hdf5') 
data1=[hist.history]
with open('./model4x44/relu/hist.pkl', 'wb') as outfile:
    pickle.dump(data1, outfile, pickle.HIGHEST_PROTOCOL)
     


















