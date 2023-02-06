# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:19:39 2021

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

#set the random seeds required for initialization
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

#normalize
inp_reno=inp_reno/100000.
inp_aoa=inp_aoa/14.0

my_inp=np.concatenate((inp_reno[:,None],inp_aoa[:,None],inp_para[:,:]),axis=1)
my_out=np.concatenate((out_cd[:,None],out_cl[:,None]),axis=1)
## Training sets
xtr0 = my_inp[I][:n]
ttr1 = my_out[I][:n]

#list of optimizers to train model
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['ftrl',
    'adam', 
    'adadelta',
    'adagrad', 
    'adamax',
    'sgd',
    'nadam',
    'rmsprop'
    ]))
# HP_L_RATE= hp.HParam('learning_rate', hp.Discrete([2.5e-4]))
METRIC_MSE = 'Mean Squared Error'

with tf.summary.create_file_writer('logs/optimizer/').as_default():
  hp.hparams_config(
    hparams=[HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_MSE, display_name='val_loss')],
  )

epochs = list(range(0,gg))

#this function sets the optimizer of the model for training based on the list given.
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
    optimizer_name = hparams[HP_OPTIMIZER]
    if optimizer_name == "adam":
        o = tf.keras.optimizers.Adam()
    elif optimizer_name == "adadelta":
        o = tf.keras.optimizers.Adadelta()
    elif optimizer_name == "adagrad":
        o = tf.keras.optimizers.Adagrad()
    elif optimizer_name == "adamax":
        o = tf.keras.optimizers.Adamax()
    elif optimizer_name == "sgd":
        o = tf.keras.optimizers.SGD()
    elif optimizer_name == "nadam":
        o = tf.keras.optimizers.Nadam()
    elif optimizer_name == "rmsprop":
        o = tf.keras.optimizers.RMSprop()
    elif optimizer_name == "ftrl":
        o = tf.keras.optimizers.Ftrl()
    else:
        raise ValueError("unexpected optimizer name: %r" % (optimizer_name))
    
    model.compile(
        optimizer=o,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
        )
  # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_name)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, mode='min',verbose=1 ,patience=100, min_lr=1.0e-8)
    history = model.fit([xtr0], [ttr1], validation_split=0.1,callbacks=[reduce_lr], epochs=gg, shuffle=False)
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
    
#tensorboard --logdir='C:/Users/lihen/projects/tf-gpu-MNIST/logs/hparam_tuning5layer'

#print details and keep logs upon script execution
for optimizer in HP_OPTIMIZER.domain.values:
    hparams = {
        HP_OPTIMIZER: optimizer
        }
    run_name = "{}".format(optimizer)
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    reset_random_seeds()
    run('logs/optimizer/' + run_name, hparams)





































