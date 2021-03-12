# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:15:11 2020

@author: lihen
"""

import matplotlib.pyplot as plt
import pandas as pd
[data] = pd.read_pickle(r'C:/Users/lihen/projects/tf-gpu-MNIST/model10x95_32/relu/hist.pkl', compression=None)
plt.figure(figsize=(6,5),dpi=1200)
plt.xscale('log')
plt.plot(data['loss'],marker='d',markerfacecolor='b',linestyle = 'None',label='train loss')
plt.plot(data['val_loss'],marker='.',markerfacecolor='r',linestyle = 'None',label='test Loss')
plt.ylim([0,0.0045])
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('Mean Squared Error',fontsize=12) 
plt.legend(loc='best',fontsize=12)
plt.savefig('./plot/ts_%s_%s.png',bbox_inches='tight', dpi=1200)
plt.show()