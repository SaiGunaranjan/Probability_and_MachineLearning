# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:24:29 2023

@author: Sai Gunaranjan
"""

""" Good reference for perceptron implementation:
    https://towardsdatascience.com/perceptron-explanation-implementation-and-a-visual-example-3c8e76b4e2d1
    """

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

plt.close('all')

Data, labels = datasets.make_classification(n_samples=200,n_features=2,n_classes=2,n_clusters_per_class=1,n_redundant=0,\
                                            class_sep=1) # random_state = 4
numDataPoints = Data.shape[0]
numFeatures = Data.shape[1]
numTrainingData = int(np.round(0.7 * numDataPoints))

labels[labels==0] -= 1
trainingData = Data[0:numTrainingData,:]
trainingLabels = labels[0:numTrainingData]

trainingDataClass0 = trainingData[trainingLabels==-1,:]
trainingDataClass1 = trainingData[trainingLabels==1,:]

testingData = Data[numTrainingData::,:]
testingLabels = labels[numTrainingData::]

testingDataClass0 = testingData[testingLabels==-1,:]
testingDataClass1 = testingData[testingLabels==1,:]



""" Perceptron training phase """
numMaxIterations = 100
wVec = np.zeros((numFeatures+1,),dtype=np.float32)
alpha = 1
for ele1 in range(numMaxIterations):
    for ele2 in range(numTrainingData):
        xVec = trainingData[ele2,:]
        xVecExt = np.hstack((xVec,1))
        wtx = np.sum(wVec * xVecExt)
        if (wtx*trainingLabels[ele2]) <= 0: # there is a small bug here
            wVec = wVec + alpha*(xVecExt * trainingLabels[ele2])


""" Testing phase"""
numTestingData = testingData.shape[0]
testingDataExt = np.hstack((testingData,np.ones((numTestingData,1))))

wtx_test = testingDataExt @ wVec
estLabels = np.zeros((testingLabels.shape),dtype=np.int32)
estLabels[wtx_test>=0] = 1
estLabels[wtx_test<0] = -1

accuracy = np.mean(estLabels == testingLabels) * 100
print('Accuracy of clasification = {0:.2f} % '.format(accuracy))

separatingLineXcordinate = np.linspace(np.amin(Data[:,0])-2,np.amax(Data[:,0])+2,100)
separatingLineYcordinate = (-wVec[-1] - wVec[0]*separatingLineXcordinate)/wVec[1]

plt.figure(1,figsize=(20,10),dpi=200)
plt.subplot(1,2,1)
plt.title('Training data')
plt.scatter(trainingDataClass0[:,0],trainingDataClass0[:,1],color='red')
plt.scatter(trainingDataClass1[:,0],trainingDataClass1[:,1],color='green')
plt.plot(separatingLineXcordinate,separatingLineYcordinate)
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
plt.plot(separatingLineXcordinate,separatingLineYcordinate)
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('y1')
plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])
