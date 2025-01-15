# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

#read data
filename='gexp1prolaris.csv'  # expression values 31 genes in 145 samples and class lables
A=pd.read_csv(filename,sep=',')
gexp=A.to_numpy()   #32 x 146 array, first columns is gene name
#gexp2=np.asarray(A) #same as the last line
labels=gexp[-1,1:]  #the last row is the class lable
gexp=gexp[:-1,1:]   #expression values  of 31 genes (row) and 145 samples (columns)
gexp=np.transpose(gexp)
gexp=gexp.astype(np.float64)
labels=labels.astype(np.float64)


filename='gexp1prolarisRef.csv' # 15 reference genes
A=pd.read_csv(filename,sep=',')
gexpref=A.to_numpy()
gexpref=gexpref[:-1,1:]
gexpref=np.transpose(gexpref)

#data statistics
#_=plt.hist(gexpref.flatten())
#plt.show()

temp=np.mean(gexpref,axis=1) #take the mean of each row (across genes/columns)
for i in range(gexp.shape[1]):
    gexp[:,i]=gexp[:,i]-temp
    
    
#_=plt.hist(gexp.flatten())
#plt.show()   
    
temp=2**(-gexp)
temp=np.mean(temp,axis=1)
CCPs=np.log2(temp)


#print([np.min(CCPs), np.max(CCPs)])
#_=plt.hist(CCPs)
#plt.show()

#creat space for (cutoff1, cutoff2) for three classe
RS=CCPs
nrepeat=20
nF=10;

minRS=np.min(RS)
maxRS=np.max(RS)
chigh=np.arange(minRS+0.2,maxRS,0.1)
cut=np.empty((2,0))
for i in range(len(chigh)):
    temp=np.arange(minRS,chigh[i]-0.1,0.1)
    temp=np.array(temp,ndmin=2)
    temp2=chigh[i]*np.ones((1,temp.shape[1]))
    temp3=np.append(temp2,temp,axis=0)  # increase length of columns
    cut=np.append(cut,temp3,axis=1) #increase length of rows
    

accuracy=np.empty(nrepeat)
recall=np.empty((nrepeat,3))
precision=np.empty((nrepeat,3))
for n in range(nrepeat):
    skf = StratifiedKFold(n_splits=nF, random_state=None,shuffle=True)
    cmatrix=np.zeros((3,3))
    for train_index, test_index in skf.split(RS, labels):
        xtrain=np.take(RS,train_index)
        ytrain=np.take(labels, train_index)
        xtest=np.take(RS,test_index)
        ytest=np.take(labels, test_index)

        err=np.empty((0))
        for j in range(cut.shape[1]):
            ycv=2*np.ones(len(ytrain))
            ycv[(xtrain>cut[0,j])!=0]=3
            ycv[(xtrain<cut[1,j])!=0]=1
            err=np.append(err,np.sum(ycv!=ytrain))    
            
        Iminerr=np.argmin(err)
        yhat=2*np.ones(len(ytest))
        yhat[(xtest>cut[0,Iminerr])!=0]=3
        yhat[(xtest<cut[1,Iminerr])!=0]=1  
        
        
        
        cmatrix=cmatrix+confusion_matrix(ytest, yhat)
       
    accuracy[n] = np.trace(cmatrix)/np.sum(cmatrix)
    for j in range(3):
        recall[n,j]=cmatrix[j,j]/np.sum(cmatrix,axis=1)[j]
        precision[n,j]=cmatrix[j,j]/np.sum(cmatrix,axis=0)[j]
        
    print(n)
    
print(cmatrix)



accuracystat=[np.mean(accuracy), np.std(accuracy)]
print(accuracystat)

recallstat=np.mean(recall,axis=0).reshape((1,3))
recallstat=np.append(recallstat,np.std(recall,axis=0).reshape(1,3), axis=0)
print(recallstat)

precisionstat=np.mean(precision,axis=0).reshape((1,3))
precisionstat=np.append(precisionstat,np.std(precision,axis=0).reshape(1,3), axis=0)
print(precisionstat)


