# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:07:35 2024

@author: Sai Gunaranjan
"""

"""
1. Larger arcitectures are very slow to run even on GPUs and the memory is huge[Resolved]
2. With samller architectures, I'm not able to hit a train/validation accuracy more than 84%

https://stackoverflow.com/questions/57984677/is-it-normal-for-gradients-to-be-extremely-large-in-a-deep-convnet
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf


Reading material for fashion MNIST dataset CNN architecture:
    https://www.kaggle.com/code/eliotbarr/fashion-mnist-tutorial
"""


import numpy as np
import tensorflow as tf
from neural_network import ConvolutionalNeuralNetwork
import time as time
import matplotlib.pyplot as plt
import cupy as cp
import sys

def load_fashion_mnist_from_csv(csv_file, numClasses):
    # Load the CSV file
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)

    # The first column contains the labels
    labels = data[:, 0].astype(int)

    # The remaining columns contain the pixel values (28x28=784)
    images = data[:, 1:].reshape(-1, 28, 28)

    """ Work with only 5 class data"""

    mask = labels < numClasses
    images = images[mask]
    labels = labels[mask]
    ##################################

    return images, labels

numClasses = 10 # Ideally the data has 10 classes

# Usage Example
input_path = r'D:\git\Probability_and_MachineLearning\datasets\fashion_mnist'
csv_file = input_path + '\\' + 'fashion-mnist_train.csv'
images_train, labels_train = load_fashion_mnist_from_csv(csv_file,numClasses)

csv_file = input_path + '\\' + 'fashion-mnist_test.csv'
images_test, labels_test = load_fashion_mnist_from_csv(csv_file,numClasses)

# Now `images` contains the image data and `labels` contains the corresponding labels
# print(f"Number of images: {images_train.shape[0]}")
# print(f"Shape of each image: {images_train.shape[1:]}")  # Should be (28, 28)
# print(f"Number of labels: {labels_train.shape[0]}")


X_train = np.transpose(images_train[:,:,:,None],(3,1,2,0))/255
Y_train = tf.keras.utils.to_categorical(labels_train,numClasses)
Y_train = Y_train.T


X_test = np.transpose(images_test[:,:,:,None],(3,1,2,0))/255
Y_test = tf.keras.utils.to_categorical(labels_test,numClasses)
Y_test = Y_test.T
# sys.exit()
numOutputNodes = Y_test.shape[0]
""" Define CNN architecture"""

""" Reference architecture from:
    1. https://www.youtube.com/watch?v=JboZfxUjLSk&list=PL1sQgSTcAaT7MbcLWacjsqoOQvqzMdUWg&index=1
    2. https://github.com/guilhermedom/cnn-fashion-mnist/blob/main/notebooks/1.0-gdfs-cnn-fashion-mnist.ipynb
    """

#(#filters, size of kernel(length), activation function)
# convLayer = [(32,3,'ReLU',1), (64,3,'ReLU',1), (64,3,'ReLU',1)] # Len of this list is the number of convolutional layers
# poolLayer = [(2,2,'maxpool'), (2,2,'maxpool'), (1,1,'maxpool')] # poolLayer = [(2,2,'maxpool'), (2,2,'maxpool'), (1,1,'maxpool')]
# denseLayer = [(250,'ReLU',1), (125,'ReLU',1), (60,'ReLU',1)] # (#nodes, activation function, whether to do BN or not) #Len of this list indicates number of hidden layers
# inputShape = (1,28,28) # Numchannles, l, w
# outputLayer = [(numOutputNodes,'softmax',0)] #(#nodes, activation function, BN set to 0 for output layer always)


""" Reference architecture from:
    https://medium.com/@sanjay_dutta/building-a-baseline-convolutional-neural-network-for-fashion-mnist-600634e5feef"""
"""
    Without BN for both CNN and DNN, train accuracy = 89%, validation accuracy = 89%, test accuracy = 89%
    With BN for both CNN and DNN, train accuracy = 86%, validation accuracy = 86%, test accuracy = 86%. But I was expecting much better results with BN enabled! Not sure why this is happening. Need to debug the reason for this. May need to add regularization and/or drop outs.
    With BN for DNN and without BN for CNN, train accuracy = 85%, validation accuracy = 85%, test accuracy = 85%. This means that BN in DNN itself is limiting the perforamance.
    With BN for CNN and without BN for DNN, train accuracy = 94%, validation accuracy = 91%, test accuracy = 91%. This implies that BN for DNN is limiting the performance and that the CNN code and BN for CNN has been implemented correctly!

"""
convLayer = [(32,3,'ReLU',1)] # Len of this list is the number of convolutional layers
poolLayer = [(2,2,'maxpool')]
denseLayer = [(100,'ReLU',0)] # #(#nodes, activation function) #Len of this list indicates number of hidden layers
inputShape = (1,28,28) # Numchannles, l, w
outputLayer = [(numOutputNodes,'softmax',0)]

""" Below architecture is same as for MNIST"""
# convLayer = [(2,5,'ReLU'), (4,3,'sigmoid')] # Len of this list is the number of convolutional layers
# poolLayer = [(2,2,'maxpool'), (2,2,'maxpool')]
# denseLayer = [] # Len of this list indicates number of hidden layers
# inputShape = (1,28,28) # Numchannles, l, w
# outputLayer = [(numOutputNodes,'softmax')]
# """ Not able to achieve more than 85% accuracy with this architecture and batchsize of 256"""

cnn = ConvolutionalNeuralNetwork(inputShape, convLayer, poolLayer, denseLayer, outputLayer)
# sys.exit()

# cnn.mlffnn.set_model_params(modeGradDescent = 'online',costfn = 'categorical_cross_entropy',epochs=100, stepsize=1e-6) # Achieved 72% accuracy with this mode
cnn.mlffnn.set_model_params(modeGradDescent = 'mini_batch',batchsize = 1024,costfn = 'categorical_cross_entropy',epochs=1000, stepsize=1e-2) #epochs=1000,stepsize=1e-3
# Since the data set is very large and also Im running a CNN, If I use smaller batch sizes,
# the time for each epoch is very very large becuase of under utilization of GPU. Hence, for the sake of
# quick results, I'm using larger batch sizes like 1024, etc for better GPU utilization and lesser compute time per epoch


split = 1 # Make it back to 1# Split data into training and testing
numDataPoints = X_train.shape[3]
numTrainingData = int(split*numDataPoints)
trainData = X_train[:,:,:,0:numTrainingData]
trainDataLabels = Y_train[:,0:numTrainingData]

""" Randomize the input while training to avoid any bias while training"""
shuffleArray = np.arange(numTrainingData)
np.random.shuffle(shuffleArray)

trainData = trainData[:,:,:,shuffleArray]
trainDataLabels = trainDataLabels[:,shuffleArray]

tstart = time.time()
# trainData = cp.asarray(trainData)
# trainDataLabels = cp.asarray(trainDataLabels)

cnn.train_cnn(trainData,trainDataLabels,split=0.8)#split=0.8


classLabels = [str(ele) for ele in range(numClasses)]
testData = X_test#X_test[:,:,:,0][:,:,:,None]#X_test
testDataLabels = Y_test#Y_test[:,0][:,None]#Y_test
numTestingData = testData.shape[3]
# testData = cp.asarray(testData)
cnn.predict_cnn(testData)
cnn.mlffnn.get_accuracy(testDataLabels, cnn.testDataPredictedLabels, printAcc=True)
cnn.mlffnn.plot_confusion_matrix(testDataLabels, cnn.testDataPredictedLabels, classLabels)

tend = time.time()

timeTrainTest = (tend - tstart)/(60*60)
print('Total time taken for training {0} examples and testing {1} examples = {2:.2f} hours'.format(numTrainingData, numTestingData, timeTrainTest))

plt.figure(2,figsize=(20,10),dpi=200)
plt.subplot(1,2,1)
plt.title('Training loss vs epochs')
plt.plot(cnn.trainingLossArray)
plt.xlabel('epochs')
plt.grid(True)

plt.subplot(1,2,2)
plt.title('Validation loss vs epochs')
plt.plot(cnn.validationLossArray)
plt.xlabel('epochs')
plt.grid(True)
