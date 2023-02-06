# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 00:04:43 2021

@author: lihen
"""

#import relevant packages and set some important parameters
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


#list of discrete batch sizes to train model
HP_BATCH= hp.HParam('batch_size', hp.Discrete([16,32,64,128,256,512]))
METRIC_MSE = 'Mean Squared Error'

with tf.summary.create_file_writer('logs/batch').as_default():
    hp.hparams_config(
    hparams=[HP_BATCH],
    metrics=[hp.Metric(METRIC_MSE, display_name='val_loss')],
  )

epochs = list(range(0,gg))

#this function sets the batch size of the model for training based on the list given.
def train_test_model(hparams):
    xx = 90
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(5,)))
    model.add(tf.keras.layers.Dense(xx, kernel_initializer='random_normal', activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(xx, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(xx, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(xx, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(xx, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(2, activation=None))
    model.summary()
    
    size = hparams[HP_BATCH] 
    if size == 16:
        b = 16
    elif size == 32:
        b = 32
    elif size == 64:
        b= 64
    elif size == 128:
        b= 128
    elif size == 256:
        b= 256
    elif size == 512:
        b= 512
    else:
        raise ValueError("unexpected size: %r" % (size))
    
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
        )
  # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_name)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=100, min_lr=1.0e-8)
    history = model.fit([xtr0], [ttr1], validation_split=0.1,callbacks=[reduce_lr], batch_size=b, epochs=gg)
    mse = np.array(history.history['val_loss'])
    return mse

#records the loss function of this model 
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    mse1 = train_test_model(hparams)
    for idx, epoch in enumerate(epochs):
        mse = mse1[idx]
        tf.summary.scalar(METRIC_MSE, mse, step=epoch+1)
    
session_num = 0
#tensorboard --logdir='C:/Users/lihen/projects/tf-gpu-MNIST/logs/hparam_tuning5layer' is the path to called Tensorboard for comparing models

#print details and keep logs upon script execution
for batch in HP_BATCH.domain.values:
    hparams = {
        HP_BATCH: batch
        }
    run_name = "%d" % (batch)
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run('logs/batch/' + run_name, hparams)
    session_num += 1
    reset_random_seeds()
     


















