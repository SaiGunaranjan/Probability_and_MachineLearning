# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:24:42 2024

@author: Sai gunaranjan Pelluri
"""

""" Disease prediction dataset kaggle link:
    https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning/data

This dataset contains features and labels for 41 diseases. I have been able to achieve an accuracy
of 97.62% with 1 hidden layer and 512 noes in the hidden layer!

    """


import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from multilayer_feedforward_nn import MLFFNeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')


""" Preprocessing for Training data"""
path = r'D:\git\Probability_and_MachineLearning\datasets\disease_prediction'
disease_data = pd.read_csv(path + '\\'+ 'Training.csv')


# Create a LabelEncoder object to convert labels to numerical values
label_encoder = preprocessing.LabelEncoder()


# Get mapping of class labels to numerical values and vice versa
classLabelsObj = label_encoder.fit(disease_data['prognosis'])
classLabels = list(classLabelsObj.classes_)
# Use the LabelEncoder object to transform the prognosis target variable
disease_data['prognosis'] = label_encoder.fit_transform(disease_data['prognosis'])


np_disease = disease_data.to_numpy()
np_disease = np_disease[:,0:-1] # To remove the nan column from the training dataset


# The input data will contain all rows and the first 132 columns
X_data = np_disease[:,0:132]

# The output data will contain all rows and the last columns
Y_data = np_disease[:,132]

# Instantiate a StandardScaler object
""" StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance.
Unit variance means dividing all the values by the standard deviation."""
scaler = StandardScaler()

# Fit the StandardScaler to the data
scaler.fit(X_data)

# Transform the input data
X_data = scaler.transform(X_data)
# print(X_data[0:5,:])
X_data = X_data.T # [NumFeatures, NumberOfFeatureVectors]


Y_data = tf.keras.utils.to_categorical(Y_data,41) # Since there are 41 features
Y_data = Y_data.T # [dimOneHotVector, NumberOfFeatureVectors]

"""Randomly shuffle data to avoid bias while fitting data"""
temp = np.hstack((X_data.T, Y_data.T))
np.random.shuffle(temp)
X_data = temp[:,0:132].T
Y_data = temp[:,132::].T

""" List of number of nodes, acivation function pairs for each layer.
1st element in architecture list is input, last element is output"""
numInputNodes = X_data.shape[0]
numOutputNodes = Y_data.shape[0]
networkArchitecture = [(numInputNodes,'Identity'), (512,'ReLU'), (numOutputNodes,'softmax')]
mlffnn = MLFFNeuralNetwork(networkArchitecture)
""" If validation loss is not changing, try reducing the learning rate"""
# mlffnn.set_model_params(modeGradDescent = 'batch',costfn = 'categorical_cross_entropy',epochs=1000, stepsize=0.0001)
# mlffnn.set_model_params(modeGradDescent = 'online',costfn = 'categorical_cross_entropy',epochs=6000, stepsize=0.0001)
"""batchsize should be a power of 2"""
mlffnn.set_model_params(modeGradDescent = 'mini_batch',batchsize = 32, costfn = 'categorical_cross_entropy',epochs=1000, stepsize=1e-3)
split = 1 # Split data into training and testing
numDataPoints = X_data.shape[1]
numTrainingData = int(split*numDataPoints)
trainData = X_data[:,0:numTrainingData]
trainDataLabels = Y_data[:,0:numTrainingData]
mlffnn.train_nn(trainData,trainDataLabels,split=0.8)


""" Preprocessing for Test data"""
path = r'D:\git\Probability_and_MachineLearning\datasets\disease_prediction'
disease_data = pd.read_csv(path + '\\'+ 'Testing.csv')

# Create a LabelEncoder object to convert labels to numerical values
label_encoder = preprocessing.LabelEncoder()


# Use the LabelEncoder object to transform the prognosis target variable
disease_data['prognosis'] = label_encoder.fit_transform(disease_data['prognosis'])
# print(disease_data.head(10))

np_disease = disease_data.to_numpy()


# The input data will contain all rows and the first 132 columns
X_data = np_disease[:,0:132]

# The output data will contain all rows and the last columns
Y_data = np_disease[:,132]

# Instantiate a StandardScaler object
""" StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance.
Unit variance means dividing all the values by the standard deviation."""
scaler = StandardScaler()

# Fit the StandardScaler to the data
scaler.fit(X_data)

# Transform the input data
X_data = scaler.transform(X_data)
# print(X_data[0:5,:])
X_data = X_data.T # [NumFeatures, NumberOfFeatureVectors]


Y_data = tf.keras.utils.to_categorical(Y_data,41) # Since there are 41 features
Y_data = Y_data.T # [dimOneHotVector, NumberOfFeatureVectors]


testData = X_data
testDataLabels = Y_data
mlffnn.predict_nn(testData)
mlffnn.get_accuracy(testDataLabels, mlffnn.testDataPredictedLabels, printAcc=True)
mlffnn.plot_confusion_matrix(testDataLabels, mlffnn.testDataPredictedLabels, classLabels)

plt.figure(2,figsize=(20,10),dpi=200)
plt.subplot(1,2,1)
plt.title('Training loss vs epochs')
plt.plot(mlffnn.trainingLossArray)
plt.xlabel('epochs')
plt.grid(True)

plt.subplot(1,2,2)
plt.title('Validation loss vs epochs')
plt.plot(mlffnn.validationLossArray)
plt.xlabel('epochs')
plt.grid(True)

