# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:40:35 2023

@author: Sai Gunaranjan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from ml_functions_lib import perceptron_train, perceptron_test, perceptron_accuracy,\
    logistic_regression_train, logistic_regression_test, logistic_regression_accuracy


plt.close('all')

""" Create the dataset for binary class"""
Data, labels = datasets.make_classification(n_samples=1000,n_features=2,n_classes=2,n_clusters_per_class=1,n_redundant=0,\
                                            class_sep=1) # random_state = 2, class_sep =1 causing issues for logistic regression
numDataPoints = Data.shape[0]
numFeatures = Data.shape[1]
numTrainingData = int(np.round(0.7 * numDataPoints))

trainingData = Data[0:numTrainingData,:]
trainingLabels = labels[0:numTrainingData]

trainingDataClass0 = trainingData[trainingLabels==0,:]
trainingDataClass1 = trainingData[trainingLabels==1,:]

testingData = Data[numTrainingData::,:]
testingLabels = labels[numTrainingData::]

testingDataClass0 = testingData[testingLabels==0,:]
testingDataClass1 = testingData[testingLabels==1,:]


""" Perceptron"""
""" Convert the labels from 0/1 to -1/1 to handle the update equation in the perceptron algorithm"""
trainingLabels_perceptron = trainingLabels.copy()
trainingLabels_perceptron[trainingLabels_perceptron==0] -= 1

testingLabels_perceptron = testingLabels.copy()
testingLabels_perceptron[testingLabels_perceptron==0] -= 1

wVec_perceptron = perceptron_train(trainingData,trainingLabels_perceptron)
estLabels_perceptron = perceptron_test(testingData,wVec_perceptron)
perceptron_accuracy(testingLabels_perceptron, estLabels_perceptron)


""" Logistic regression"""
wVec_logReg, logLikelihood = logistic_regression_train(trainingData,trainingLabels)
estLabels_logReg = logistic_regression_test(testingData,wVec_logReg)
logistic_regression_accuracy(testingLabels, estLabels_logReg)


""" Plotting the separating hyper plane"""
separatingLineXcordinate = np.linspace(np.amin(Data[:,0])-2,np.amax(Data[:,0])+2,100)
separatingLineYcordinate_perceptron = (-wVec_perceptron[-1] - wVec_perceptron[0]*separatingLineXcordinate)/wVec_perceptron[1]
separatingLineYcordinate_logReg = (-wVec_logReg[-1] - wVec_logReg[0]*separatingLineXcordinate)/wVec_logReg[1]


plt.figure(1,figsize=(20,10),dpi=200)
plt.suptitle('Perceptron')
plt.subplot(1,2,1)
plt.title('Training data')
plt.scatter(trainingDataClass0[:,0],trainingDataClass0[:,1],color='red')
plt.scatter(trainingDataClass1[:,0],trainingDataClass1[:,1],color='green')
plt.plot(separatingLineXcordinate,separatingLineYcordinate_perceptron)
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('y1')
plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])
# plt.axis('scaled')

plt.subplot(1,2,2)
plt.title('Testing data')
plt.scatter(testingDataClass0[:,0],testingDataClass0[:,1],color='red')
plt.scatter(testingDataClass1[:,0],testingDataClass1[:,1],color='green')
plt.plot(separatingLineXcordinate,separatingLineYcordinate_perceptron)
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('y1')
plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])


plt.figure(2,figsize=(20,10),dpi=200)
plt.suptitle('Logistic Regression')
plt.subplot(1,2,1)
plt.title('Training data')
plt.scatter(trainingDataClass0[:,0],trainingDataClass0[:,1],color='red')
plt.scatter(trainingDataClass1[:,0],trainingDataClass1[:,1],color='green')
plt.plot(separatingLineXcordinate,separatingLineYcordinate_logReg)
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('y1')
plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])
# plt.axis('scaled')

plt.subplot(1,2,2)
plt.title('Testing data')
plt.scatter(testingDataClass0[:,0],testingDataClass0[:,1],color='red')
plt.scatter(testingDataClass1[:,0],testingDataClass1[:,1],color='green')
plt.plot(separatingLineXcordinate,separatingLineYcordinate_logReg)
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('y1')
plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])


plt.figure(3,figsize=(20,10),dpi=200)
plt.title('Logistic regression: Log likelihood vs iterations')
plt.plot(logLikelihood[1::],'-o')
plt.xlabel('Iterations')
plt.grid(True)
