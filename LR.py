# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:50:52 2020

@author: lihen
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:15:11 2020

@author: lihen
"""

import matplotlib.pyplot as plt
import pandas as pd
plt.figure(figsize=(6,5),dpi=1200)
plt.xscale('log')
[data] = pd.read_pickle(r'C:/Users/lihen/projects/tf-gpu-MNIST/model/hist.pkl', compression=None)
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('RMSE',fontsize=12) 
plt.plot(data['lr'],marker='.', markerfacecolor='b', color='b',markersize=1,label='learning rate')
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('Learning Rate',fontsize=12) 
plt.legend(loc='best',fontsize=12)
plt.savefig('./plot/learningrate.png',bbox_inches='tight', dpi=1200)
plt.show()