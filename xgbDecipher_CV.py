

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier


#read data
filename='gexp3decipher.csv'  # expression values 31 genes in 145 samples and class lables
A=pd.read_csv(filename,sep=',')
gexp=A.to_numpy()   #32 x 146 array, first columns is gene name
#gexp2=np.asarray(A) #same as the last line
labels=gexp[-1,1:]  #the last row is the class lable
gexp=gexp[:-1,1:]   #expression values  of 31 genes (row) and 145 samples (columns)
gexp=np.transpose(gexp)
gexp=gexp.astype(np.float64)
#labels=labels.astype(np.float64)
labels=labels.astype(np.int16)
labels = labels - 1

    
    
#_=plt.hist(gexp.flatten())
#plt.show()   
    

nrepeat=10
nF=10
cmatrices=np.empty((nrepeat,3,3))
accuracy=np.empty(nrepeat)
recall=np.empty((nrepeat,3))
precision=np.empty((nrepeat,3))
bestparams=np.empty((nrepeat,4))

params = { 'max_depth': [6], 
          'learning_rate': [0.05, 0.06, 0.07],
          'reg_lambda': [1.8, 2, 2.5],
          'n_estimators': [750, 800, 850]}


t1=time.time()

for n in range(nrepeat):  
    # cross valication
    xgbc=XGBClassifier()
    clf = GridSearchCV(estimator=xgbc,
                    param_grid=params,
                    #scoring='mlogloss',
                    #scoring='accuracy'
                    cv=nF,
                    verbose=0)
    clf.fit(gexp,labels)                      
    
    #print("Best parameters:", clf.best_params_)
    #print("Highest score: ", clf.best_score_)
        
    best_max_depth=clf.best_params_['max_depth']
    best_learning_rate=clf.best_params_['learning_rate']
    best_n_estimators=clf.best_params_['n_estimators']
    best_reg_lambda=clf.best_params_['reg_lambda']
    
    bestparams[n,0]= best_max_depth
    bestparams[n,1]= best_learning_rate
    bestparams[n,2]= best_n_estimators
    bestparams[n,3]= best_reg_lambda
    
    # CV error at the optimal parameter values
    skf = StratifiedKFold(n_splits=nF, random_state=None,shuffle=True)
    cmatrix=np.zeros((3,3))
    for train_index, test_index in skf.split(gexp, labels):
        xtrain=np.take(gexp,train_index, axis=0)
        ytrain=np.take(labels, train_index)
        xtest=np.take(gexp,test_index, axis=0)
        ytest=np.take(labels, test_index)
    
        xgbc=XGBClassifier(max_depth=best_max_depth,learning_rate=best_learning_rate,
                           n_estimators=best_n_estimators,reg_lambda=best_reg_lambda)
    
        xgbc.fit(xtrain,ytrain)
        ypred=xgbc.predict(xtest)
        cmatrix=cmatrix+confusion_matrix(ytest,ypred)  
    
    print(cmatrix)
    cmatrices[n,:,:]=cmatrix
    accuracy[n] = np.trace(cmatrix)/np.sum(cmatrix)
    for j in range(3):
        recall[n,j]=cmatrix[j,j]/np.sum(cmatrix,axis=1)[j]
        precision[n,j]=cmatrix[j,j]/np.sum(cmatrix,axis=0)[j]
        

t2=time.time()
print(t2-t1)


#compute results statistics        
accuracystat=[np.mean(accuracy), np.std(accuracy)]
accuracystat=np.array(accuracystat)
print(accuracystat)

recallstat=np.mean(recall,axis=0).reshape((1,3))
recallstat=np.append(recallstat,np.std(recall,axis=0).reshape(1,3), axis=0)
print(recallstat)

precisionstat=np.mean(precision,axis=0).reshape((1,3))
precisionstat=np.append(precisionstat,np.std(precision,axis=0).reshape(1,3), axis=0)
print(precisionstat)

print(cmatrices)
print(bestparams)
        
#save results

accuracystat=np.reshape(accuracystat, (2, -1))
accuracystat=np.append(accuracystat,np.zeros((2,2)),axis=1)
aprstat=np.concatenate((accuracystat,recallstat,precisionstat),axis=0)
filename='xgb_decipher_aprstat.csv'
with open(filename, 'w', newline="") as file:
    csvwriter = csv.writer(file)    
    csvwriter.writerows(aprstat)
    
accuracy=np.reshape(accuracy,(nrepeat,-1))
accuracy=np.append(accuracy,np.zeros((nrepeat,2)),axis=1)
apr=np.concatenate((accuracy, precision, recall), axis=1)
filename='xgb_decipher_apr.csv'
with open(filename, 'w', newline="") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerows(apr)

cmatrix=cmatrices[0,:,:]
for i in range(nrepeat-1):
    cmatrix=np.append(cmatrix,cmatrices[i+1,:,:],axis=0)
filename='xgb_decipher_cmatrix.csv'
with open(filename, 'w', newline="") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerows(cmatrix)
    
filename='xgb_decipher_best_params.csv'
with open(filename, 'w', newline="") as file:
    csvwriter = csv.writer(file)    
    csvwriter.writerows(bestparams)



