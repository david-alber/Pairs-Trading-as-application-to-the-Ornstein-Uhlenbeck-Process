# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:57:23 2020

@author: alber
"""

import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import coint

import slib as slb
plt.close('all')
#%%Load data
start_year = 2010; end_year = 2020;
names = np.load('names.npy'); symbols = np.load('symbols.npy'); 
variation = np.load('variation.npy'); close_val = np.load('close_val.npy'); open_val = np.load('open_val.npy')

#%%Cointegration Value
# =============================================================================
# scores, pvalues, pairs = slb.find_cointegrated_pairs(close_val)
# #Save data
# np.save('scores',scores)
# np.save('pvalues',pvalues)
# np.save('pairs',pairs)
# =============================================================================
pairs = np.load('pairs.npy'); pvalues = np.load('pvalues.npy'); scores = np.load('scores.npy'); 
slb.covplot(pvalues,symbols,'Cointegration')

pvalue = slb.cointplt(close_val,pairs[1],names,start_year,end_year,plot=True)#1#5#9
#%%
S1 = pairs[1,0]; S2 = pairs[1,1]
ratio = close_val[S1]/close_val[S2]
H=slb.hurstcoeff(close_val,pairs[1],names,plot=True)






