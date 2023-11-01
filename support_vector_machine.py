# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:30:10 2023

@author: Sai Gunaranjan
"""


import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

plt.close('all')

class SVM:

    def __init__(self, trainingData, testingData, trainingLabels):

        self.trainingData = trainingData;
        self.testingData = testingData;
        self.trainingLabels = trainingLabels;
        self.numTrainingData = self.trainingData.shape[0]
        self.Y = np.diag(self.trainingLabels)

        return

    def svm_train(self):

        """ SVM training phase """

        """ Cap the maximum number of iterations of the gradient ascent step for solving the dual formulation
        variable alpha."""
        numMaxIterations = 1000#500

        """ The dataset might have a bias and need not always be around the origin. It could be shifted.
        For example, the data might be separated by the line wTx = -5. We do not know the bias apriori. So,
        to handle this, we do the following.
        We include the unknown bias also as another parameter to the w vector. We then also append a 1 to the feature vector
        ([x, 1]). Hence, the dimensionality of both the feature vector and the parameter vector w are increased by 1.
        """

        Y = np.diag(self.trainingLabels)
        oneVector = np.ones((self.numTrainingData,))
        trainingDataExt = np.hstack((self.trainingData,np.ones((self.numTrainingData,1)))) # Appending 1s to the training data.
        # X = trainingDataExt.T
        X = self.trainingData.T
        kernel = ((X.T @ X) + 1)**2# Polynomial kernel
        Iminus11T = np.eye(self.numTrainingData) - ((1/self.numTrainingData) * (oneVector[:,None] @ oneVector[None,:]))
        meanRemovedKernal = kernel#Iminus11T.T @ kernel @ Iminus11T # kernel
        """ Choice of hyper parameter c: Articles
        https://www.baeldung.com/cs/ml-svm-c-parameter#:~:text=Selecting%20the%20Optimal%20Value%20of%20C&text=depends%20on%20the%20specific%20problem,training%20error%20and%20margin%20width.
        """
        c = 10#10#0.05#40000
        self.alphaVec = np.zeros((self.numTrainingData,),dtype=np.float32)
        self.costFunctionDualProblem = np.zeros((numMaxIterations,),dtype=np.float32)
        for ele1 in range(numMaxIterations):
            self.costFunctionDualProblem[ele1] = (self.alphaVec @ oneVector) - (0.5 * (self.alphaVec[None,:] @ Y.T @ (meanRemovedKernal) @ Y @ self.alphaVec[:,None]))
            eta = 1/((ele1+1))#1e-4#1/((ele1+1)**2) # Learning rate/step size. Typically set as 1/t or 1/t**2
            gradient = oneVector - (Y.T @ (meanRemovedKernal) @ Y @ self.alphaVec) # Y.T @ (X.T @ X) @ Y can be moved outside the loop
            self.alphaVec = self.alphaVec + eta*gradient
            self.alphaVec[self.alphaVec<0] = 0 # box constraints. alpha should always be >=0
            self.alphaVec[self.alphaVec>c] = c
            # print('Min alpha val = {0:.5f}, Max alpha val = {1:.5f}'.format(np.amin(self.alphaVec),np.amax(self.alphaVec)))
        # Remove belowline
        # wVec_svm = X @ Y @ self.alphaVec # This is the actual wVec but it doesnt need to be explicitly computed
        suppVecIndMargin = np.where((self.alphaVec>0) & (self.alphaVec<c))[0]
        print('Number of supporting vectors = {}'.format(len(suppVecIndMargin)))

        """ Reference for the below method of bias calculation is given in:
            1. https://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-training-algorithms/#:~:text=Once%20training%20has%20been%20completed%2C%20we%20can%20calculate%20the%20bias%20b
            2. https://stats.stackexchange.com/questions/362046/how-is-bias-term-calculated-in-svms#:~:text=Generally%20speaking%20the%20bias%20term%20is%20calculated"""
        # bias = np.mean(trainingLabels[suppVecIndMargin] - (wVec_svm[0:-1][None,:] @ X[0:-1,suppVecIndMargin]))
        # bias = np.mean(trainingLabels[suppVecIndMargin] - (wVec_svm @ X[:,suppVecIndMargin]))
        # wVec_svm = np.hstack((wVec_svm,bias))
        """ There's some issue in the bias computation. Need to resolve this"""

        return



    def svm_test(self):

        """ Testing phase"""
        numTestingData = self.testingData.shape[0]
        testingDataExt = np.hstack((self.testingData,np.ones((numTestingData,1)))) # Appending 1s to the test data as well.
        wtx_test = self.decision_function(self.testingData)

        estLabels = np.zeros((wtx_test.shape),dtype=np.int32)
        estLabels[wtx_test>=0] = 1
        estLabels[wtx_test<0] = -1

        return estLabels


    def decision_function(self,X):

        train = self.trainingData.T # d x n
        test = X.T # d x k
        numEvalPoints = test.shape[1]
        kernel = ((train.T @ test) + 1)**2# Polynomial kernel
        oneVector_train = np.ones((self.numTrainingData,))
        oneVector_test = np.ones((numEvalPoints,))
        Iminus11T_train = np.eye(self.numTrainingData) - ((1/self.numTrainingData) * (oneVector_train[:,None] @ oneVector_train[None,:]))
        Iminus11T_test = np.eye(numEvalPoints) - ((1/numEvalPoints) * (oneVector_test[:,None] @ oneVector_test[None,:]))
        meanRemovedKernal = kernel#Iminus11T_train.T @ kernel @ Iminus11T_test # kernel

        """ wVec = X @ Y @ alpha"""
        """ wVec*X = alpha.T @ Y.T @ XTX = alpha.T @ Y.T @ K where K is the kernel matrix"""
        wtx_test = self.alphaVec[None,:] @ self.Y.T @ meanRemovedKernal
        wtx_test = wtx_test.squeeze()

        return wtx_test


    def decision_boundary(self):

        plt.scatter(self.trainingData[:, 0], self.trainingData[:, 1], c=self.trainingLabels, s=50, cmap=plt.cm.Paired, alpha=.5)
        ax = plt.gca()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors=['b', 'g', 'r'], levels=[-1, 0, 1], alpha=0.5,
                        linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])

        # highlight the support vectors
        ax.scatter(self.trainingData[:, 0][self.alphaVec > 0.], self.trainingData[:, 1][self.alphaVec > 0.], s=50,
                   linewidth=1, facecolors='none', edgecolors='k')

    def svm_accuracy(self,testingLabels, estLabels):

        accuracy = np.mean(estLabels == testingLabels) * 100
        print('\nAccuracy of clasification SVM = {0:.2f} % \n'.format(accuracy))



Data, labels = datasets.make_circles(n_samples=500, noise=0.05, random_state=None, factor=0.5)
# Data, labels =  datasets.make_moons(n_samples=500, noise=0.05, random_state=6) For this dataset, kernel has to change from polynomial to gaussian/radial basis function


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

""" Support Vector machine"""
trainingLabels_svm = trainingLabels.copy()
trainingLabels_svm[trainingLabels_svm==0] -= 1


testingLabels_svm = testingLabels.copy()
testingLabels_svm[testingLabels_svm==0] -= 1


""" Remove bias/mean in the data"""
# trainingData -= np.mean(trainingData,axis=0)[None,:]
# testingData -= np.mean(testingData,axis=0)[None,:]
svm_obj = SVM(trainingData, testingData, trainingLabels_svm)
svm_obj.svm_train()
estLabels_svm = svm_obj.svm_test()
svm_obj.svm_accuracy(testingLabels_svm, estLabels_svm)
svm_obj.decision_boundary()


plt.figure(5,figsize=(20,10),dpi=200)
plt.title('SVM: Dual problem cost function vs iterations')
plt.plot(svm_obj.costFunctionDualProblem[1::],'-o')
plt.xlabel('Iterations')
plt.grid(True)