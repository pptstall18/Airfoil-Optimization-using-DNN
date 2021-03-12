# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:58:43 2021

@author: lihen
"""

import time
start_time = time.time()
import random
import tensorflow as tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
from tensorboard.plugins.hparams import api as hp
# Python 3.5
import numpy as np
import pandas as pd
import sys
from tensorflow.keras.callbacks import ReduceLROnPlateau
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
result = pd.read_pickle('./naca4_clcd_turb_st_3para.pkl', compression=None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1615400000)
   tf.random.set_seed(1615400000)
   np.random.seed(1615400000)
   random.seed(1615400000)
gg=1
layers = 10
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

HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['sigmoid',
                                                     'relu',
                                                     'tanh',
                                                     'softmax',
                                                     'softplus',
                                                     'softsign',
                                                     'selu',
                                                     'elu',
                                                     'exponential']))
METRIC_MSE = 'Mean Squared Error'

with tf.summary.create_file_writer('logs/activation').as_default():
    hp.hparams_config(
    hparams=[HP_ACTIVATION],
    metrics=[hp.Metric(METRIC_MSE, display_name='val_loss')],
    )
    
epochs = list(range(0,gg))

def train_test_model(hparams):
    xx = 30
    activation_name = hparams[HP_ACTIVATION]
    if activation_name == "sigmoid":
        a = tf.keras.activations.sigmoid
    elif activation_name == "relu":
        a = tf.keras.activations.relu
    elif activation_name == "tanh":
        a = tf.keras.activations.tanh
    elif activation_name == "softmax":
        a = tf.keras.activations.softmax
    elif activation_name == "softplus":
        a = tf.keras.activations.softplus
    elif activation_name == "selu":
        a = tf.keras.activations.selu
    elif activation_name == "elu":
        a = tf.keras.activations.elu
    elif activation_name == "exponential":
        a = tf.keras.activations.exponential
    else:
        raise ValueError("unexpected optimizer name: %r" % (activation_name))
    model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(5,)),
    tf.keras.layers.Dense(xx, kernel_initializer='random_normal', activation=a),
    tf.keras.layers.Dense(xx, activation=a),
    tf.keras.layers.Dense(2, activation=None),
    ])
    
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2.5e-4),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
        )
  # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_name)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=100, min_lr=1.0e-8)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2.5e-4), loss='mean_squared_error')
    history = model.fit([xtr0], [ttr1], validation_split=0.1,callbacks=[reduce_lr], epochs=gg)
    mse = np.array(history.history['val_loss'])
    return mse

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    mse1 = train_test_model(hparams)
    for idx, epoch in enumerate(epochs):
        mse = mse1[idx]
        tf.summary.scalar(METRIC_MSE, mse, step=epoch+1)
    
session_num = 0
#tensorboard --logdir='C:/Users/lihen/projects/tf-gpu-MNIST/logs/hparam_tuning5layer'

for activation in HP_ACTIVATION.domain.values:
    hparams = {
        HP_ACTIVATION: activation
        }
    run_name = "{}".format(activation)
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run('logs/activation/' + run_name, hparams)
    session_num += 1
    reset_random_seeds()
 


















