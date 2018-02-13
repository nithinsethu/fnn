# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:15:50 2018

@author: Nithin
"""
import Iris as iris
import numpy as np
import ann_mod2 as amod
ts = 150
y = np.zeros((3,ts))
#print(np.shape(y[:,0]))
for i in range(ts):
    y[:,i] = iris.getLabel(iris.labels[i])
    
 
    
    
t = amod.ANN(np.transpose(iris.features),y,[4,3,3],70)
t.train()
t.test()
        
        