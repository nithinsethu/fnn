import pandas as pd
import numpy as np
#def getLabel(s):
#    if(s=='Iris-setosa'):
#        return np.array([(1,),(0,),(0,)])
#    elif(s=='Iris-versicolor'):
#        return np.array([(0,),(1,),(0,)])
#    elif(s=='Iris-virginica'):
#        return np.array([(0,),(0,),(1,)])
def getLabel(s):
    if(s=='Iris-setosa'):
        return np.array([1,0,0])
    elif(s=='Iris-versicolor'):
        return np.array([0,1,0])
    elif(s=='Iris-virginica'):
        return np.array([0,0,1])

df = pd.read_csv('iris.csv',header=None)
data = df.as_matrix()
m = np.shape(data)[1]-1
n = np.shape(data)[0]
np.random.shuffle(data)
f = data[:,0:m].copy()
features =  np.array(list(f), dtype=np.float64)
m = m+1
labels = data[:,m-1].copy()
#print(features)
