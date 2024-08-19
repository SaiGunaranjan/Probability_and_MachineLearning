# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:05:47 2024

@author: Sai Gunaranjan
"""

"""
I have implemented my first neural network architecture and trained the weights using the back propagation algorithm.
My implementation is based on Shastry's NPTEL lecture series Pattern Recognition, lecture number 25, 26, 27, 28'

Derivation of the output of softmax function (vector input , vector output)
wrt input to softmax function
https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1

I have also derived based on the above link. The derivation is availabe in my OneNote at:
    Machine Learning notes tab, Backpropagation algorithm page


Multi class classification using softmax and cross entropy.



Reference:
https://medium.com/@ja_adimi/neural-networks-101-hands-on-practice-25df515e13b0

"""

"""
NN for classification problem

1. Inititalize all the weights. How to initialize W matrices of each layer
2. Normalize input features to zero mean and unit variance, so that no one features totally dominates the output.
3. Add momentum term to the gradient descent algo
4. Variable learning rate/step size i.e large step size initially and smaller step size as we progress over more iterations
5. Batch vs online vs mini batch mode of gradient descent
6. Cross validation with validation dataset (k fold cross validation)
7. How are Accuracy and loss curves computed on the validation dataset
"""

import numpy as np
np.random.seed(0)

class MLFFNeuralNetwork():

    def __init__(self):

        # layerType = 'input'
        # activationFunction = 'identity'
        numFeatures = 5 #example
        self.numNodesLayer1 = numFeatures #numNodesInputLayer

        """ Initialize the weights matrix to 0s. Other initializations like picking from a normal distribution are also possible"""
        self.numNodesLayer2 = 128
        self.weightMatrixLayer1to2 = np.zeros((self.numNodesLayer2,self.numNodesLayer1+1),dtype=np.float32) # +1 is for the bias term

        self.numNodesLayer3 = 128
        self.weightMatrixLayer2to3 = np.zeros((self.numNodesLayer3,self.numNodesLayer2+1),dtype=np.float32) #+1 is for the bias term

        self.numNodesLayer4 = 3 # numNodesOutputLayer #output is a 1 hot vector of 3 classes
        self.weightMatrixLayer3to4 = np.zeros((self.numNodesLayer4,self.numNodesLayer3+1),dtype=np.float32) #+1 is for the bias term



    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def tanh(self,z):
        # tanh = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        ePowpz = np.exp(z)
        ePowmz = np.exp(-z)
        return (ePowpz - ePowmz) / (ePowpz + ePowmz)

    def softmax(self,z):
        """ z has to be a vector"""
        ePowz = np.exp(z)
        return ePowz/np.sum(ePowz)

    def ReLU(self,z):
        return np.maximum(0,z)

    def sigmoid_derivative(self,z):
        return self.sigmoid(z) * (1-self.sigmoid(z))

    def tanh_derivative(self,z):
        return 1 - (self.tanh(z)**2)

    def ReLU_derivative(self,z):
        return 1*(z>0) # Returns 0 for negative values, 1 for posotive values

    def forwardpass(self):
        pass

    def backwardpass(self):
        pass

    def train_nn(self,trainData,trainDataLabels):
        # trainDataLabels should also be a 1 hot vector representation
        self.backpropagation(trainData,trainDataLabels, mode = 'online',costfn = 'categorical_cross_entropy',epochs=10)

    def backpropagation(self,trainData,trainDataLabels, mode = 'online',costfn = 'categorical_cross_entropy',epochs=10):

        stepSize = 0.01#
        numTrainData = trainData.shape[1]
        # numFeatures = trainData.shape[0]
        arr = np.arange(numTrainData)
        for ele1 in np.arange(epochs):
            np.random.shuffle(arr) # Randomly shuffle the order of feeding the training data for each epoch
            for ele2 in arr:



                # Forward pass (put into a function)
                inputVector = np.ones((self.numNodesLayer1+1),dtype=np.float32) # +1 is to account for the bias term
                inputVector[1::] = trainData[:,ele2] # 1st element of each data vector is 1 and the weight corresponding to 1 is the bias term
                outputVectorLayer1 = inputVector # y1 = x

                inputVectorLayer2 = self.weightMatrixLayer1to2 @ outputVectorLayer1
                outputVectorLayer2 = self.ReLU(inputVectorLayer2) # Activation function for layer 2

                temp = outputVectorLayer2
                outputVectorLayer2 = np.ones((self.numNodesLayer2+1),dtype=np.float32) # +1 is to account for the bias term
                outputVectorLayer2[1::] = temp # 1st element of each data vector is 1 and the weight corresponding to 1 is the bias term
                inputVectorLayer3 = self.weightMatrixLayer2to3 @ outputVectorLayer2
                outputVectorLayer3 = self.ReLU(inputVectorLayer3) # Activation function for layer 3

                temp = outputVectorLayer3
                outputVectorLayer3 = np.ones((self.numNodesLayer3+1),dtype=np.float32) # +1 is to account for the bias term
                outputVectorLayer3[1::] = temp
                inputVectorLayer4 = self.weightMatrixLayer3to4 @ outputVectorLayer3
                outputVectorLayer4 = self.softmax(inputVectorLayer4) # Finaly layer output

                costFunction = -np.sum((trainDataLabels[:,ele2]*np.log2(outputVectorLayer4))) # -Sum(di*log(yi)), where di is the actual output and yi is the predicted output

                # Backward pass (put into function)
                # trainDataLabels should also be a 1 hot vector representation
                errorLayer4 = outputVectorLayer4 - trainDataLabels[:,ele2] # (y - d) For softmax function with categorical cross entropy cost function
                errorLayer3 = (self.weightMatrixLayer3to4[:,1::].T @ errorLayer4) * self.ReLU_derivative(inputVectorLayer3) # Derivative of the activation function at layer 3 evaluated at input vector at layer 3(nl)
                errorLayer2 = (self.weightMatrixLayer2to3[:,1::].T @ errorLayer3) * self.ReLU_derivative(inputVectorLayer2) # Derivative of the activation function at layer 2 evaluated at input vector at layer 2(nl)

                # Errors/delta obtained for all the layers from 2 to L
                # Compute gradient wrt Wl_ij
                # dJ/dWl_ij = deltal+1_j * yi_l
                # gradient # dJ/dwij for all i,j
                self.weightMatrixLayer1to2 = self.weightMatrixLayer1to2 - stepSize*(errorLayer2[:,None] @ outputVectorLayer1[None,:])
                self.weightMatrixLayer2to3 = self.weightMatrixLayer2to3 - stepSize*(errorLayer3[:,None] @ outputVectorLayer2[None,:])
                self.weightMatrixLayer3to4 = self.weightMatrixLayer3to4 - stepSize*(errorLayer4[:,None] @ outputVectorLayer3[None,:])





mlffnn = MLFFNeuralNetwork()
trainData = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
trainDataLabels = np.array([[1,0],[0,1],[0,0]])
mlffnn.train_nn(trainData,trainDataLabels)














