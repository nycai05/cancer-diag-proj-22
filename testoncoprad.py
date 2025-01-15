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
filename='gexp2oncotypeDx.csv'  # expression values 12 genes in 145 samples and class lables
A=pd.read_csv(filename,sep=',')
gexp=A.to_numpy()   #32 x 146 array, first columns is gene name
#gexp2=np.asarray(A) #same as the last line
labels=gexp[-1,1:]  #the last row is the class lable
gexp=gexp[:-1,1:]   #expression values  of 31 genes (row) and 145 samples (columns)
gexp=np.transpose(gexp)
gexp=gexp.astype(np.float64)
labels=labels.astype(np.float64)


filename='gexp2oncotypeDxRef.csv' # 15 reference genes
A=pd.read_csv(filename,sep=',')
gexpref=A.to_numpy()
gexpref=gexpref[:-1,1:]
gexpref=np.transpose(gexpref)
#gexpref=gexpref.astype(np.float64)

#data statistics
#_=plt.hist(gexp.flatten())
#plt.show()



#genes
#AZGP1,BGN,COL1A1,FAM13C,FLNC,GSN,GSTM2,KLK2,SFRP4,SRD5A2,TPM2, TPX2
#step 1

temp=np.mean(gexpref,axis=1) #take the mean of each row (across genes/columns)
for i in range(gexp.shape[1]):
    gexp[:,i]=gexp[:,i]-temp
gexp=gexp+10
    
#_=plt.hist(gexp.flatten())
#plt.show()   
    

temp=gexp[:,9] #SRD5A2
#_=plt.hist(temp)
#plt.show()
temp[temp<5.5]=5.5
gexp[:,9]=temp


temp=gexp[:,11] #TPX2
temp[temp<5]=5
gexp[:,11]=temp


#genes
#AZGP1,BGN,COL1A1,FAM13C,FLNC,GSN,GSTM2,KLK2,SFRP4,SRD5A2,TPM2, TPX2
#step 2
stromalindex=np.array([2, 3, 9])-1 # BGN, CoL1A1, SFRP4
stromal=(0.527*gexp[:,stromalindex[0]]+0.457*gexp[:,stromalindex[1]]
        +0.156*gexp[:,stromalindex[2]])

cellularindex=np.array([5, 6, 11, 7])-1   #FLNC, GSN, TMP2, GSTM2
cellular=0.163*gexp[:,cellularindex[0]]+0.504*gexp[:,cellularindex[1]] \
        +0.421*gexp[:,cellularindex[2]]+0.394*gexp[:,cellularindex[3]]
    
Androgenindex=np.array([4, 8, 1, 10])-1 #FAM13C, KLK2, AZGP1, SRD5A2
Androgen=0.634*gexp[:,Androgenindex[0]] + 1.079*gexp[:,Androgenindex[1]] \
        + 0.642*gexp[:,Androgenindex[2]]+ 0.997*gexp[:,Androgenindex[3]]
    
    
Proliferation=gexp[:,11] #TPX2

#step 3
GPSu=0.735*stromal - 0.368*cellular - 0.352*Androgen + 0.095*Proliferation

#step 4
GPS=13.4*(GPSu-10.5); #a low GPS implies low risk, class 1 is high risk, 
                        #class 3 is low risk
    

    
#print([np.min(GPS), np.max(GPS)])
#_=plt.hist(GPS)
#plt.show()

m=(np.min(GPS)+np.max(GPS))/2
GPS=GPS-m+50


RS=GPS
RS[RS<0]=0
RS[RS>100]=100

#creat space for (cutoff1, cutoff2) for three classe
nrepeat=20
nF=10;

minRS=np.ceil(np.min(RS))+1
maxRS=np.floor(np.max(RS))
chigh=np.arange(minRS+20,maxRS-1,1)

cut=np.empty((2,0))
for i in range(len(chigh)):
    temp=np.arange(minRS,chigh[i],1)
    temp=np.array(temp,ndmin=2)
    temp2=chigh[i]*np.ones((1,temp.shape[1]))
    temp3=np.append(temp2,temp,axis=0)  # increase number of rows
    cut=np.append(cut,temp3,axis=1) #increase number of columns
    

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
            ycv[(xtrain>cut[0,j])!=0]=1
            ycv[(xtrain<cut[1,j])!=0]=3
            err=np.append(err,np.sum(ycv!=ytrain))    
            
        Iminerr=np.argmin(err)
        yhat=2*np.ones(len(ytest))
        yhat[(xtest>cut[0,Iminerr])!=0]=1
        yhat[(xtest<cut[1,Iminerr])!=0]=3  
        
        
        
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


