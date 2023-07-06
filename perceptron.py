# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:24:29 2023

@author: Sai Gunaranjan
"""

""" Good reference for perceptron implementation:
    https://towardsdatascience.com/perceptron-explanation-implementation-and-a-visual-example-3c8e76b4e2d1

Perceptron algorithm

In this script, I have implemented the perceptron algorithm. Perceptron algorithm is a supervised learning algorithm for
 binary classification when the data is linearly separable with a "gamma margin separability". The implementation is based on the video lectures of Arun Rajkumar and
the blog:
https://towardsdatascience.com/perceptron-explanation-implementation-and-a-visual-example-3c8e76b4e2d1

For generating the 2 class data/labels, I have used the "make_classification" function of the datasets class imported from
the sklearn package. Here we can mention the number of data samples, dimensionality of the feature vector, number of classes,
number of clusters per class, amount of separation across the classes and so on. This is a very useful function for data generation.
The perceptron algorithm converges when the training data is perfectly linearly separable with a gamma margin i.e
wTx must not just be greater than zero but is should be greater than 0 by a gamma margin. In other words,
the perceptron algorithm converges if wTx*y >= gamma , for all x and a positive gamma. So,
linear separability is only a necessary condition but not a sufficient condition. But when the training data is mixed
i.e. not perfectly linearly separable, then the perceptron algorithm does not converge. This can be theoretically proved and
the proof is given in the lectures of Arun Rajkumar.  But with actual real data, we cannot always ensure that the training data
is linearly separable (or with a gamma margin). Hence, the perceptron algorithm might not converge at all.
So in order to implement the perceptron algorithm even in such cases, we need to cap the maximum number
of iterations that the perceptron algorithm runs for, else it will never converge. So, even in the code, I have fixed
the maximum number of iterations for the percpetron algorithm. Theoretically, we can show that the time/number of iterations for
the perceptron algorithm to converge is proportional to the radius of the farthest data point and inversely proportional to
the gamma separation/margin between the closest points between the two classes. The closer the points of the two classes,
the more time the percpetron algorithm takes to converge.

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

plt.close('all')

""" Create the dataset for binary class"""
Data, labels = datasets.make_classification(n_samples=1000,n_features=2,n_classes=2,n_clusters_per_class=1,n_redundant=0,\
                                            class_sep=1) # random_state = 4
numDataPoints = Data.shape[0]
numFeatures = Data.shape[1]
numTrainingData = int(np.round(0.7 * numDataPoints))

""" Convert the labels from 0/1 to -1/1 to handle the update equation in the perceptron algorithm"""
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

""" Cap the maximum number of iterations of the perceptron algorithm. Ideally this should be a function of radius of
farthest point in the dataset and the gamma margin of separation between the 2 classes."""
numMaxIterations = 100

""" The dataset might have a bias and need not always be around the origin. It could be shifted.
For example, the data might be separated by the line wTx = -5. But the perceptron algorithm is defined assuming data
doesn't have any bias. So, just checking for sign(wTx) to get the class labels will not be correct. We need to check if wTx >= -5 or wTx <-5.
Also, we do not know the bias apriori. So, to handle this, we do the following.
We include the unknown bias also as another parameter to the w vector. We then also append a 1 to the feature vector ([x, 1]). Hence,
the dimensionality of both the feature vector and the parameter vector w are increased by 1.
"""
wVec = np.zeros((numFeatures+1,),dtype=np.float32) # +1 to take care of the bias in the data
alpha = 1 # Added a momentum parameter but set it to 1 for now.
for ele1 in range(numMaxIterations):
    for ele2 in range(numTrainingData):
        xVec = trainingData[ele2,:]
        xVecExt = np.hstack((xVec,1)) # Appending 1 to the feature vector as well.
        wtx = np.sum(wVec * xVecExt)
        if (wtx*trainingLabels[ele2]) <= 0: # The equal sign is to handle the initialization of w to 0 and hence an update is required at the first iteration itself.
            wVec = wVec + alpha*(xVecExt * trainingLabels[ele2]) # Update equation for the perceptron algorithm : W_k+1 = W_k + x*y


""" Testing phase"""
numTestingData = testingData.shape[0]
testingDataExt = np.hstack((testingData,np.ones((numTestingData,1)))) # Appending 1s to the test data as well.

wtx_test = testingDataExt @ wVec
estLabels = np.zeros((testingLabels.shape),dtype=np.int32)
estLabels[wtx_test>=0] = 1
estLabels[wtx_test<0] = -1

accuracy = np.mean(estLabels == testingLabels) * 100
print('Accuracy of clasification = {0:.2f} % '.format(accuracy))

""" Plotting the separating hyper plane"""
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
