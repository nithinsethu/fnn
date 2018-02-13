# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:34:24 2018

@author: 16pt23
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:01:24 2018

@author: Nithin
"""

import numpy as np


class ANN:
    def __init__(self,x,y,no_nodes,train_pct,alpha=0.5):
        n = self.no_training_examples = int((np.shape(x)[1])*train_pct/100)
        m = self.no_testing_examples = np.shape(x)[1] - self.no_training_examples
        self.x_train = x[:,:self.no_training_examples].copy()
        self.y_train = y[:,:self.no_training_examples].copy()
        self.x_test = x[:,self.no_training_examples:].copy()
        self.y_test = y[:,self.no_training_examples:].copy()
        self.no_layers = len(no_nodes) - 1
        self.no_nodes = no_nodes
        self.weights = []
        self.bias = []
        
        self.alpha = alpha
        for i in range(self.no_layers):
            self.weights.append(np.random.rand(self.no_nodes[i+1],self.no_nodes[i])*0.01)
            self.bias.append(np.random.rand(self.no_nodes[i+1],1)*0.01)
#            self.weights.append(np.random.uniform(low=0, high=1, size=(self.no_nodes[i+1],self.no_nodes[i])))
#            self.bias.append(np.random.uniform(low=0, high=1, size=(self.no_nodes[i+1],1)))

            
    

            
    def __sig(self,z,der = False):
        if(der):
            return self.__sig(z)*(1-self.__sig(z))
        else:
            return 1/(1+np.exp(-z))
    
    
    
    def train(self):
        n = self.no_training_examples
        l = self.no_layers
        for j in range(10000):
            y=[]
            z=[]
            delta=[]
            y.append(self.x_train.copy())
            
            for i in range(l):
                z.append(np.matmul(self.weights[i],y[i])+self.bias[i].copy())
                y.append(self.__sig(z[i])) 
                 
            delta.append((self.y_train-y[l])*self.__sig(z[l-1],True))
            for i in range(l-1,0,-1):
                delta.append(np.matmul(np.transpose(self.weights[i]),delta[l-1-i])*self.__sig(z[i-1],True))
            delta.reverse()
            for i in range(l):
                self.weights[i] = self.weights[i] + self.alpha*np.matmul(delta[i],np.transpose(y[i]))/n
                self.bias[i] = self.bias[i] + self.alpha*np.matmul(delta[i],np.ones((n,1)))/n
    
    
    
    
    def test(self):
        l = self.no_layers
        n = self.no_testing_examples
        y=[]
        z=[]
        y.append(self.x_test.copy())
        for i in range(l):
            z.append(np.matmul(self.weights[i],y[i])+self.bias[i].copy())
            y.append(self.__sig(z[i]))
        res = y[l]
        count =0
        mat = self.genConfMatrix(np.round(res))
        print(self.metrics(mat))
    def genConfMatrix(self,y):
        n = self.no_nodes[self.no_layers]
        m = np.shape(y)[1]
        mat = np.zeros((n,n))
        o =  self.y_test
        x = self.unique(np.transpose(self.y_test)).copy()
        d = {}
        for i in range(n):
            arr = np.reshape(x[:,i],(n,1))
            arr = tuple(map(tuple, arr))
            d[arr] = i
        for i in range(m):
            arr = np.reshape(y[:,i],(n,1))
            arr2 = np.reshape(o[:,i],(n,1))
            arr = tuple(map(tuple, arr))
            arr2 = tuple(map(tuple, arr2))
#            print(arr2,arr)
            mat[d[arr2],d[arr]] +=1
        return mat
    def metrics(self,mat):
        n = len(mat)
        acc = 0
        s = sum(sum(mat))
        for i in range(n):
            acc += mat[i,i]
        acc = acc/s
        return acc
            
            
    def unique(self,a):
        order = np.lexsort(a.T)
        a = a[order]
        diff = np.diff(a, axis=0)
        ui = np.ones(len(a), 'bool')
        ui[1:] = (diff != 0).any(axis=1) 
        return a[ui]
            

            













#            change = np.matmul(delta[0],np.transpose(y[0]))/n
#            #print(y[1][:,0],change)
#            #input()
#            self.bias[0] = self.bias[0] +  self.alpha*np.matmul(delta[0],np.ones((n,1)))/n
#            self.weights[0] = self.weights[0] + self.alpha*change
#        count = 0 
#        for i in range(150):
#            res = self.__sig(np.matmul(self.weights[0],self.x_test)[:,i] + np.reshape(self.bias[0],(1,3)))
#            #print(np.shape(res),np.shape(np.reshape(self.y_test[:,i],(1,3))))
#            if((np.round(res) == np.reshape(self.y_test[:,i],(1,3))).all()):
#                count+=1
#        print(count)
