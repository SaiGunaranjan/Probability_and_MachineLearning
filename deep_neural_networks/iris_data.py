# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:17:39 2024

@author: Sai Gunaranjan
"""

""" Reference for data preprocessing steps like feature normalization, one hot vector creation, etc:
    https://medium.com/@ja_adimi/neural-networks-101-hands-on-practice-25df515e13b0

    In the above blog, they were able to achieve an accuracy of 93% using keras and tensorflow.
    I was able to achieve 96% accuracy with my own NN functions and architecture.
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from multilayer_feedforward_nn import MLFFNeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

path = r'D:\git\Probability_and_MachineLearning\datasets'
iris_data = pd.read_csv(path + '\\'+ 'iris.csv')
# print(iris_data.head(10))

# Create a LabelEncoder object
label_encoder = preprocessing.LabelEncoder()

# Use the LabelEncoder object to transform the Species target variable
iris_data['Species'] = label_encoder.fit_transform(iris_data['Species'])
# print(iris_data.head(10))

np_iris = iris_data.to_numpy()
np_iris = np_iris[:,1::]
# print(np_iris[:5])

# The input data will contain all rows and the first 4 columns
X_data = np_iris[:,0:4]

# The output data will contain all rows and the last columns
Y_data = np_iris[:,4]

# Instantiate a StandardScaler object
scaler = StandardScaler()

# Fit the StandardScaler to the data
scaler.fit(X_data)

# Transform the input data
X_data = scaler.transform(X_data)
# print(X_data[0:5,:])
X_data = X_data.T # [NumFeatures, NumberOfFeatureVectors]


Y_data = tf.keras.utils.to_categorical(Y_data,3)
Y_data = Y_data.T # [dimOneHotVector, NumberOfFeatureVectors]

"""Randomly shuffle data to avoid bias while fitting data"""
temp = np.hstack((X_data.T, Y_data.T))
np.random.shuffle(temp)
X_data = temp[:,0:4].T
Y_data = temp[:,4::].T

""" List of number of nodes, acivation function pairs for each layer.
1st element in architecture list is input, last element is output"""
numInputNodes = X_data.shape[0]
numOutputNodes = Y_data.shape[0]
networkArchitecture = [(numInputNodes,'Identity'), (128,'ReLU'), (128, 'ReLU'), (numOutputNodes,'softmax')]
mlffnn = MLFFNeuralNetwork(networkArchitecture)
""" If validation loss is not changing, try reducing the learning rate"""
mlffnn.set_model_params(modeGradDescent = 'batch',costfn = 'categorical_cross_entropy',epochs=1000, stepsize=0.0001)
split = 0.8 # Split data into training and testing
numDataPoints = X_data.shape[1]
numTrainingData = int(split*numDataPoints)
trainData = X_data[:,0:numTrainingData]
trainDataLabels = Y_data[:,0:numTrainingData]
mlffnn.train_nn(trainData,trainDataLabels,split=0.8)

testData = X_data[:,numTrainingData::]
testDataLabels = Y_data[:,numTrainingData::]
mlffnn.predict_nn(testData)
predClasses = np.argmax(mlffnn.testDataPredictedLabels,axis=0)
actualClasses = np.argmax(testDataLabels,axis=0)
accuracy = np.mean(predClasses == actualClasses) * 100
print('\nAccuracy of NN = {0:.2f} % \n'.format(accuracy))

plt.figure(1,figsize=(20,10),dpi=200)
plt.subplot(1,2,1)
plt.title('Training loss vs epochs')
plt.plot(mlffnn.costFunctionArray)
plt.xlabel('epochs')
plt.grid(True)

plt.subplot(1,2,2)
plt.title('Validation loss vs epochs')
plt.plot(mlffnn.validationLossArray)
plt.xlabel('epochs')
plt.grid(True)

