# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:09:19 2024

@author: Sai Gunaranjan
"""

""" In this script, I have implemented the XOR gate both as a classification as well as a
regression problem using multi layered feed forward NN. I have tested using both the online and batch
mode of gradient descent methods."""

from multilayer_feedforward_nn import MLFFNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

""" If modeClassification is True, NN will be used for a classifcation
and if modeClassification is False, NN will be used for regression """
modeClassification = True

""" If modeOnlineGradientDes is True, Online/Stochastic mode of gradient descent will be used
and if modeOnlineGradientDes is False, Batch mode of gradient descent will be used """
modeOnlineGradientDes = True


if modeClassification == True:

    """ NN for classification"""

    X_data = np.array([[0,0],[0,1],[1,0],[1,1]]).T
    Y_data = np.array([[1,0],[0,1],[0,1],[1,0]]).T # One hot vector
    """ List of number of nodes, acivation function pairs for each layer.
    1st element in architecture list is input, last element is output"""
    numInputNodes = X_data.shape[0]
    numOutputNodes = Y_data.shape[0]
    networkArchitecture = [(numInputNodes,'Identity'), (2, 'sigmoid'), (2,'softmax')]
    mlffnn = MLFFNeuralNetwork(networkArchitecture)
    if modeOnlineGradientDes == True:
        mlffnn.set_model_params(modeGradDescent = 'online',costfn = 'categorical_cross_entropy',epochs=10000, stepsize=0.1)
    else:
        mlffnn.set_model_params(modeGradDescent = 'batch',costfn = 'categorical_cross_entropy',epochs=10000, stepsize=0.1)
    trainData = X_data
    trainDataLabels = Y_data
    mlffnn.train_nn(trainData,trainDataLabels)
    mlffnn.predict_nn(trainData,trainDataLabels)
    mlffnn.testDataPredictedLabels[mlffnn.testDataPredictedLabels>=0.5] = 1
    mlffnn.testDataPredictedLabels[mlffnn.testDataPredictedLabels<0.5] = 0
    print('\nActual labels', trainDataLabels[1,:])
    print('\n Predicted labels', mlffnn.testDataPredictedLabels[1,:])
    numTrainingSamples = trainData.shape[1]
    plt.figure(1,figsize=(20,10),dpi=200)
    plt.title('Cost / loss function')
    if modeOnlineGradientDes == True:
        plt.plot(mlffnn.costFunctionArray[0::numTrainingSamples])
    else:
        plt.plot(mlffnn.costFunctionArray)
    plt.xlabel('Epochs')
    plt.grid(True)

else:

    """ NN for regression"""

    X_data = np.array([[0,0],[0,1],[1,0],[1,1]]).T
    Y_data = np.array([0,1,1,0])[None,:]
    """ List of number of nodes, acivation function pairs for each layer.
    1st element in architecture list is input, last element is output"""
    numInputNodes = X_data.shape[0]
    numOutputNodes = Y_data.shape[0]
    networkArchitecture = [(numInputNodes,'Identity'), (2, 'sigmoid'), (1,'sigmoid')]
    mlffnn = MLFFNeuralNetwork(networkArchitecture)
    if modeOnlineGradientDes == True:
        mlffnn.set_model_params(modeGradDescent = 'online',costfn = 'squared_error',epochs=100000,stepsize=0.1)
    else:
        mlffnn.set_model_params(modeGradDescent = 'batch',costfn = 'squared_error',epochs=10000,stepsize=0.1)
    trainData = X_data
    trainDataLabels = Y_data
    mlffnn.train_nn(trainData,trainDataLabels)
    mlffnn.predict_nn(trainData,trainDataLabels)
    mlffnn.testDataPredictedLabels[mlffnn.testDataPredictedLabels>=0.5] = 1
    mlffnn.testDataPredictedLabels[mlffnn.testDataPredictedLabels<0.5] = 0
    print('\nActual labels', trainDataLabels[0,:])
    print('\n Predicted labels', mlffnn.testDataPredictedLabels[0,:])
    numTrainingSamples = trainData.shape[1]
    plt.figure(2,figsize=(20,10),dpi=200)
    plt.title('Cost / loss function')
    if modeOnlineGradientDes == True:
        plt.plot(mlffnn.costFunctionArray[0::numTrainingSamples])
    else:
        plt.plot(mlffnn.costFunctionArray)
    plt.xlabel('Epochs')
    plt.grid(True)