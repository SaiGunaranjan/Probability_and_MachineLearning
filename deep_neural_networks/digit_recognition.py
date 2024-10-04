# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:11:52 2024

@author: Sai Gunaranjan
"""

""" In this script, I have performed digit classification of MNIST dataset
(https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and achieved 90% accuracy
on the test data. By trial and erro, I found that an ANN architecture with 1 hidden layer
with 100 neurons and batch mode of GD with stepsize of 1e-6 is required to train the model to achive
90% traning accuracy, validation accuracy and testing accuracy.
Even though, I have given 10000 epochs, the model achieves 90% accuracy after about 1541 epochs.

The input images are 8 bit quantized values and have values ranging from 0 to 255.
One important aspect I learned in this exercise is that, we need to normalize the input image
by 255 to ensure the input data lies between 0 to 1. Normlaizing this way, helps ensure stability
of the weights.



Need to go from ANN to CNN?
https://www.quora.com/Why-do-we-use-CNN-when-we-already-have-ANN-with-a-fully-connected-structure
1. Exploding number of parameters/weights especially when the size of input image is very large
2. Larger the input size, more the number of weights/parameters --> more number of examples
required to train the network.
3. For an ANN, an image with cat at top left corner of image is different from an image with cat at
bottom right of the image, so it treats it as two different outputs. Whereas, a CNN does a
local weighting/activation and hence for a CNN, both the images are treated the same.

Hence we will move to CNNS for image datasets.

"""

#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import tensorflow as tf
from multilayer_feedforward_nn import MLFFNeuralNetwork
import matplotlib.pyplot as plt


plt.close('all')
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = np.zeros((size,rows,cols))
        # for i in range(size):
        #     images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28) # each image shape is 28 x 28
            images[i,:,:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)


#
# Verify Reading Dataset via MnistDataloader class
#

import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = r'D:\git\Probability_and_MachineLearning\datasets\mnist'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);
        index += 1

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#
# Show some random training and test images
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

# show_images(images_2_show, titles_2_show)


X_data = ((x_train.reshape(60000,28*28)).T)/255
Y_data = np.array(y_train)
Y_data = tf.keras.utils.to_categorical(Y_data,10) # Since there are 28*28 features
Y_data = Y_data.T # [dimOneHotVector, NumberOfFeatureVectors]
""" List of number of nodes, acivation function pairs for each layer.
1st element in architecture list is input, last element is output"""
numInputNodes = X_data.shape[0]
numOutputNodes = Y_data.shape[0]
networkArchitecture = [(numInputNodes,'Identity'), (100,'ReLU'), (numOutputNodes,'softmax')]
mlffnn = MLFFNeuralNetwork(networkArchitecture)
""" If validation loss is not changing, try reducing the learning rate"""
mlffnn.set_model_params(modeGradDescent = 'batch',costfn = 'categorical_cross_entropy',epochs=10000, stepsize=1e-6)
# mlffnn.set_model_params(modeGradDescent = 'online',costfn = 'categorical_cross_entropy',epochs=6000, stepsize=0.0001)
"""batchsize should be a power of 2"""
# mlffnn.set_model_params(modeGradDescent = 'mini_batch',batchsize = 2048, costfn = 'categorical_cross_entropy',epochs=1000, stepsize=1e-1)
split = 1 # Split data into training and testing
numDataPoints = X_data.shape[1]
numTrainingData = int(split*numDataPoints)
trainData = X_data[:,0:numTrainingData]
trainDataLabels = Y_data[:,0:numTrainingData]
mlffnn.train_nn(trainData,trainDataLabels,split=0.8)

X_data = ((x_test.reshape(10000,28*28)).T)/255
Y_data = np.array(y_test)
Y_data = tf.keras.utils.to_categorical(Y_data,10) # Since there are 28*28 features
Y_data = Y_data.T # [dimOneHotVector, NumberOfFeatureVectors]
classLabels = [str(ele) for ele in range(10)]
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
