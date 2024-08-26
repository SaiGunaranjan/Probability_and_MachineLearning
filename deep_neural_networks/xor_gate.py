# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:09:19 2024

@author: Sai Gunaranjan
"""

from multilayer_feedforward_nn import MLFFNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# """ NN for classification"""
# X_data = np.array([[0,0],[0,1],[1,0],[1,1]]).T
# Y_data = np.array([[1,0],[0,1],[0,1],[1,0]]).T

# """ List of number of nodes, acivation function pairs for each layer.
# 1st element in architecture list is input, last element is output"""
# numInputNodes = X_data.shape[0]
# numOutputNodes = Y_data.shape[0]
# networkArchitecture = [(numInputNodes,'Identity'), (2, 'sigmoid'), (2,'softmax')]
# mlffnn = MLFFNeuralNetwork(networkArchitecture)
# mlffnn.set_model_params(mode = 'online',costfn = 'categorical_cross_entropy',epochs=100000, stepsize=0.1)
# trainData = X_data
# trainDataLabels = Y_data
# mlffnn.train_nn(trainData,trainDataLabels)
# mlffnn.predict_nn(trainData,trainDataLabels)
# print('\nActual labels', trainDataLabels[1,:])
# print('\n Predicted labels', mlffnn.testDataPredictedLabels[1,:])

""" NN for regression"""
X_data = np.array([[0,0],[0,1],[1,0],[1,1]]).T
Y_data = np.array([0,1,1,0])[None,:]

""" List of number of nodes, acivation function pairs for each layer.
1st element in architecture list is input, last element is output"""
numInputNodes = X_data.shape[0]
numOutputNodes = Y_data.shape[0]
networkArchitecture = [(numInputNodes,'Identity'), (2, 'sigmoid'), (1,'sigmoid')]
mlffnn = MLFFNeuralNetwork(networkArchitecture)
mlffnn.set_model_params(mode = 'online',costfn = 'squared_error',epochs=100000,stepsize=0.1)
trainData = X_data
trainDataLabels = Y_data
mlffnn.train_nn(trainData,trainDataLabels)
mlffnn.predict_nn(trainData,trainDataLabels)
mlffnn.testDataPredictedLabels[mlffnn.testDataPredictedLabels>=0.5] = 1
mlffnn.testDataPredictedLabels[mlffnn.testDataPredictedLabels<0.5] = 0
print('\nActual labels', trainDataLabels[0,:])
print('\n Predicted labels', mlffnn.testDataPredictedLabels[0,:])

numTrainingSamples = trainData.shape[1]


plt.figure(1,figsize=(20,10),dpi=200)
plt.title('Cost / loss function')
plt.plot(mlffnn.costFunctionVal[0::numTrainingSamples])
plt.xlabel('Epochs')
plt.grid(True)