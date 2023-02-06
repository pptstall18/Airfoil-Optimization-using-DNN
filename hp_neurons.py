# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 00:01:05 2021

@author: lihen
"""

#import relevant packages and set some important parameters
import time
start_time = time.time()
import numpy as np
import random
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
# Python 3.5
import os
import pandas as pd
import sys
from tensorflow.keras.callbacks import ReduceLROnPlateau
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_rows", None, "display.max_columns", None)
result = pd.read_pickle(r'C:/Users/lihen/projects/tf-gpu-MNIST/naca4_clcd_turb_st_3para.pkl', compression=None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
layers = [1,2,3,4,5,6,7,8,9,10]

#set the random seeds required for initialization
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1615400000)
   tf.random.set_seed(1615400000)
   np.random.seed(1615400000)
   random.seed(1615400000)
   os.environ['TF_DETERMINISTIC_OPS'] = '1'
gg=1000

#load data from naca4_clcd_turb_st_3para.pkl
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

#normalize the numeral values such that the max value is 1
inp_reno=inp_reno/100000.
inp_aoa=inp_aoa/14.0

my_inp=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
my_out=np.concatenate((out_cd[:,None],out_cl[:,None]),axis=1)
## Training sets
xtr0 = my_inp[I][:n]
ttr1 = my_out[I][:n]

#Hyperparameters set from 1 to 10 layers with neurons in each layer in multiples of 10, up to 100
HP_NEURONS = hp.HParam('neurons', hp.Discrete(range(10,101,10)))
HP_LAYERS = hp.HParam('layer', hp.Discrete([1,2,3,4,5,6,7,8,9,10]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_L_RATE= hp.HParam('learning_rate', hp.Discrete([2.5e-4]))
METRIC_MSE = 'Mean Squared Error'

with tf.summary.create_file_writer('logs/1 hidden layer').as_default():
    hp.hparams_config(
    hparams=[HP_LAYERS, HP_NEURONS, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_MSE, display_name='val_loss')],
  )

epochs = list(range(0,gg))

#this function builds the layers and neurons of the model for training based on the list given.
def train_test_model(hparams, layer,rundir):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(5,)))
    model.add(tf.keras.layers.Dense(hparams[HP_NEURONS], kernel_initializer='random_normal', activation=tf.keras.activations.relu))
    if layer > 1: 
        for i in range(layer-1):
             model.add(tf.keras.layers.Dense(hparams[HP_NEURONS], activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(2, activation=None))
    model.summary()
    optimizer_name = hparams[HP_OPTIMIZER]
    learning_rate = hparams[HP_L_RATE]
    if optimizer_name == "adam":
        o = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        o = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError("unexpected optimizer name: %r" % (optimizer_name,))
    model.compile(
        optimizer=o,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
        )
    # logdir=rundir
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch = 100000000)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=0 ,patience=100, min_lr=1.0e-8)
    history = model.fit([xtr0], [ttr1], validation_split=0.1,callbacks=[reduce_lr], epochs=gg,shuffle=False)
    mse = np.array(history.history['val_loss'])
    return mse

#records the loss function of this model 
def run(run_dir, hparams, layer):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        mse1 = train_test_model(hparams, layer, run_dir)
        for idx, epoch in enumerate(epochs):
            mse = mse1[idx]
            tf.summary.scalar(METRIC_MSE, mse, step=epoch+1)
    
#tensorboard --logdir='C:/Users/lihen/projects/tf-gpu-MNIST/logs/hparam_tuning5layer'
for layer in HP_LAYERS.domain.values:
    for neurons in HP_NEURONS.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for learning_rate in HP_L_RATE.domain.values:
                hparams = {
                    HP_LAYERS: layer,
                    HP_NEURONS: neurons,
                    HP_OPTIMIZER: optimizer,
                    HP_L_RATE: learning_rate
                    }
                run_name = "{}".format(neurons) + " Neurons"
                print('--- Starting trial: %s' % run_name)
                print("{} layer ".format(layer))
                print({h.name: hparams[h] for h in hparams})
                reset_random_seeds()
                run('logs/{} hidden layer '.format(layer) + run_name, hparams, layer)





