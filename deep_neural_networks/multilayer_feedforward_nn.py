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


Multi class classification using softmax and categorical cross entropy.
Regression using sigmoid/tanh and squared error loss function

Updates:

05/08/2024
The script can now create a user defined NN architecture
Following are the features
1. Can program the number of hidden layers, number of nodes per hidden layer, activation function for each layer. All passed as a list. List of #nodes, activation function tuples. Length of the list indicates the total number of layers of the neural network. The entries of the first and last elements of the list denote the input layer and output layer respectively.

2. Defined a function named 'set_model_params' where we can decide the following:
    1. Type of gradient descent for the training process like batch, online/stochastic, mini batch, etc.
    2. Cost function to optimize i.e categorical cross entropy, squared error loss, etc. Typically, we use categorical cross entropy for classification tasks and squared error loss furnction for regression tasks
    3. Number of epochs of the training

3. Broke down the backpropagation function into sub functions namely:
    1. Forward pass
    2. Cost function evaluation post forward pass
    3. Backward pass
    4. Update weights

4. Both forward pass and backward pass can now can cater to any number of user defined layers in the network through a for loop

28/08/2024
5. Brought up both the online/stochastic as well as batch mode of gradient descent.

6. Online mode of gradient descent requires smaller step sizes and more number of epochs since weights get updated with each data point and each epoch

7. Batch mode of gradient descent does not require such small step sizes and fewer number of epochs since weights get updated less frequenctly (once per epoch)



Reference:
https://medium.com/@ja_adimi/neural-networks-101-hands-on-practice-25df515e13b0

"""

"""
NN for classification problem

1. Inititalize all the weights. How to initialize W matrices of each layer
2. Normalize input features to zero mean and unit variance, so that no one features totally dominates the output.
3. Add momentum term to the gradient descent algo
4. Variable learning rate/step size i.e large step size initially and smaller step size as we progress over more iterations
5. Batch vs online vs mini batch mode of gradient descent. [Done. Mini batch mode pening]
6. Cross validation with validation dataset (k fold cross validation)
7. How are Accuracy and loss curves computed on the validation dataset
8. Make provision for batch mode and mini batch mode of training as well.[Done. Mini batch mode pening]
9. Clean up multiple calls of the is else for the activation and derivative of activation function. [Done]
10. Compare my implementation of the forward pass with tensorflow/pytorch implementation [Done]
11. Check whether backward pass is correct.[Done]
12. Check if the weights update step is correct.[Done. It is correct]
13. Cost/loss function is not changing at all with epochs! [Done. I just had to increase the number of epochs from 10k to 100k for the cost function to converge and come close to 0]
14. Remove multiple passing of the 'stepSize' parameter to several functions! [Done]
"""

import numpy as np
np.random.seed(0)

class MLFFNeuralNetwork():

    def __init__(self,networkArchitecture):

        self.networkArchitecture = networkArchitecture
        """ weights initialization between each successive layers"""
        self.numLayers = len(self.networkArchitecture)
        self.weightMatrixList = []
        for ele in range(self.numLayers-1):
            """Weight matrix from layer l to layer l+1 """
            numNodesLayerL = self.networkArchitecture[ele][0]
            numNodesLayerLplus1 = self.networkArchitecture[ele+1][0]
            """ Initialize the weights matrix to 0s. Other initializations like picking from a normal distribution are also possible"""
            # Initialization of weights with 0s doesnt seem to be convering for the stochastic gradient descent method!
            # weightMatrix = np.zeros((numNodesLayerLplus1,numNodesLayerL+1),dtype=np.float32) # +1 is for the bias term
            """ Initialize the weights matrix to random uniformly drawn values from 0 to 1"""
            weightMatrix = np.random.rand(numNodesLayerLplus1,numNodesLayerL+1) # +1 is for the bias term
            self.weightMatrixList.append(weightMatrix)

    def set_model_params(self,modeGradDescent = 'online',costfn = 'categorical_cross_entropy',epochs = 100000, stepsize = 0.1):
        self.modeGradDescent = modeGradDescent
        self.costfn = costfn
        self.epochs = epochs
        self.stepsize = stepsize



    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def tanh(self,z):
        # tanh = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        ePowpz = np.exp(z)
        ePowmz = np.exp(-z)
        return (ePowpz - ePowmz) / (ePowpz + ePowmz)

    def ReLU(self,z):
        return np.maximum(0,z)

    def softmax(self,z):
        """ z has to be a vector"""
        ePowz = np.exp(z)
        return ePowz/np.sum(ePowz,axis=0)[None,:]



    def sigmoid_derivative(self,z):
        return self.sigmoid(z) * (1-self.sigmoid(z))

    def tanh_derivative(self,z):
        return 1 - (self.tanh(z)**2)

    def ReLU_derivative(self,z):
        return 1*(z>0) # Returns 0 for negative values, 1 for positive values

    def activation_function(self, itaLayerL, activationFn):
        if (activationFn == 'sigmoid'):
            layerLOutput = self.sigmoid(itaLayerL)
        elif (activationFn == 'tanh'):
            layerLOutput = self.tanh(itaLayerL)
        elif (activationFn == 'ReLU'):
            layerLOutput = self.ReLU(itaLayerL)
        elif (activationFn == 'softmax'):
            layerLOutput = self.softmax(itaLayerL)

        return layerLOutput

    def derivative_activation_function(self, itaLayerL, activationFn):
        if (activationFn == 'sigmoid'):
            activationFnDerivative = self.sigmoid_derivative(itaLayerL)
        elif (activationFn == 'tanh'):
            activationFnDerivative = self.tanh_derivative(itaLayerL)
        elif (activationFn == 'ReLU'):
            activationFnDerivative = self.ReLU_derivative(itaLayerL)

        return activationFnDerivative


    def compute_loss_function(self,trainDataLabel):

        if (self.costfn == 'categorical_cross_entropy'):
            costFunction = -np.sum((trainDataLabel*np.log2(self.predictedOutput))) # -Sum(di*log(yi)), where di is the actual output and yi is the predicted output
        elif (self.costfn == 'squared_error'):
            costFunction = 0.5*np.sum((self.predictedOutput - trainDataLabel)**2) # 1/2 Sum((yi - di)**2), where di is the actual output and yi is the predicted output

        return costFunction


    def forwardpass(self, trainDataSample):
        # Below is just a hack line to check if the forward pass is correctly codes given the correct weights

        # self.weightMatrixList = [np.array([[-0.5,  1. ,  1. ],
        #                                    [-1.5,  1. ,  1. ]]),
        #                          np.array([[-0.5,  1. ,  -2. ]])]

        # self.weightMatrixList = [np.array([[-5.6836409,  3.70978057,  3.71140367],
        #                                     [ -2.42011388,  5.78819908, 5.79615577 ]]),
        #                           np.array([[-3.37817248,   -8.14903896 ,  7.50503069 ]])]
        """ Forward pass"""
        layerLOutput = trainDataSample
        numTrainingSamples = trainDataSample.shape[1]
        self.Ita = []
        self.outputEachlayer = []
        # ele3 is looping over the layers
        for ele3 in range(self.numLayers):
            numNodesLayerL = self.networkArchitecture[ele3][0]
            if (ele3 == 0):
                """Input layer"""
                layerLminus1Output = np.ones((numNodesLayerL + 1, numTrainingSamples),dtype=np.float32) # +1 is to account for the bias term
                layerLminus1Output[1::,:] = layerLOutput
            else:
                weightMatrixLayerLminus1toL = self.weightMatrixList[ele3-1]
                itaLayerL = weightMatrixLayerLminus1toL @ layerLminus1Output
                activationFn = self.networkArchitecture[ele3][1] # Activation function name
                layerLOutput = self.activation_function(itaLayerL,activationFn) # gives output of the activation function for the ita input

                layerLminus1Output = np.ones((numNodesLayerL + 1, numTrainingSamples),dtype=np.float32) # +1 is to account for the bias term
                layerLminus1Output[1::,:] = layerLOutput

                self.Ita.append(itaLayerL) # ita is not stored for input layer. It is stored for all other layers.
            self.outputEachlayer.append(layerLminus1Output) # Output for each layer
        self.predictedOutput = layerLOutput



    def backwardpass(self, trainDataLabel):
        # This backward pass computation is for 'categorical_cross_entropy', 'squared error' loss functions and online and batch mode. Need to add the computation of backward pass for mini batch

        # numTrainingSamples = trainDataLabel.shape[1]
        self.errorEachLayer = []
        # ele4 loop goes from layer L-1(output) to layer 0 input
        for ele4 in range(self.numLayers-1,0,-1):
            if (ele4 == self.numLayers-1):
                """ Final output layer"""
                if (self.costfn == 'categorical_cross_entropy'):
                    errorLayerL = self.predictedOutput - trainDataLabel # (y - d) For softmax activation function with categorical cross entropy cost function. Used for classificcation tasks.
                elif (self.costfn == 'squared_error'):
                    itaLayerL = self.Ita[ele4-1]
                    activationFn = self.networkArchitecture[ele4][1]
                    activationFnDerivative = self.derivative_activation_function(itaLayerL,activationFn)
                    errorLayerL = (self.predictedOutput - trainDataLabel) * activationFnDerivative # (y - d)f'(ita_L) For any other activation function(other than softmax) with squared error cost function. Used for regression tasks
                errorLayerLplus1 = errorLayerL
            else:
                weightMatrixLayerLtoLplus1 = self.weightMatrixList[ele4]
                itaLayerL = self.Ita[ele4-1]
                activationFn = self.networkArchitecture[ele4][1]
                activationFnDerivative = self.derivative_activation_function(itaLayerL,activationFn)
                errorLayerL = (weightMatrixLayerLtoLplus1[:,1::].T @ errorLayerLplus1) * activationFnDerivative # Derivative of the activation function at layer 3 evaluated at input vector at layer 3(nl)
                errorLayerLplus1 = errorLayerL

            self.errorEachLayer.append(errorLayerLplus1) # These error arrays are packed from layer L-1 down to 1 and not from 1 to L-1. They are arranged in reverse order. (layers start from 0 to L-1)
        # Errors/delta obtained for all the layers from 2 to L



    def update_weights(self):
        # Errors/delta obtained for all the layers from 2 to L
        # Compute gradient wrt Wl_ij
        # dJ/dWl_ij = deltal+1_j * yi_l
        # gradient # dJ/dwij for all i,j
        count = -1
        for ele4 in range(self.numLayers-1):
            gradientCostFnwrtWeights = self.errorEachLayer[count] @ self.outputEachlayer[ele4].T
            self.weightMatrixList[ele4] = self.weightMatrixList[ele4] - self.stepsize*gradientCostFnwrtWeights
            count -= 1


    def train_nn(self,trainData,trainDataLabels):
        # trainDataLabels should also be a 1 hot vector representation for classification task
        self.backpropagation(trainData,trainDataLabels)


    def stochastic_gradient_descent(self,trainData,trainDataLabels, arr):
        """ arr is the randomly shuffled order of sampling the training data"""
        for ele2 in arr:
            trainDataSample = trainData[:,ele2][:,None]
            trainDataLabel = trainDataLabels[:,ele2][:,None]

            """ Forward pass"""
            self.forwardpass(trainDataSample)

            """ Cost function computation"""
            self.costFunctionValue = self.compute_loss_function(trainDataLabel) # Keep appending the cost function value across data points and epochs
            self.costFunctionArray.append(self.costFunctionValue)
            """ Backward pass"""
            self.backwardpass(trainDataLabel)

            """ Update weights"""
            self.update_weights()


    def batch_gradient_descent(self,trainData,trainDataLabels):

        trainDataSample = trainData
        trainDataLabel = trainDataLabels

        """ Forward pass"""
        self.forwardpass(trainDataSample)

        """ Cost function computation"""
        self.costFunctionValue = self.compute_loss_function(trainDataLabel) # Keep appending the cost function value across epochs
        self.costFunctionArray.append(self.costFunctionValue)

        """ Backward pass"""
        self.backwardpass(trainDataLabel)

        """ Update weights"""
        self.update_weights()



    def backpropagation(self,trainData,trainDataLabels):
        """ Currently the back propagation is coded for online and batch mode of weight update. Need to add for mini batch mode as well"""

        numTrainData = trainData.shape[1]
        # numFeatures = trainData.shape[0]
        arr = np.arange(numTrainData)
        self.costFunctionArray = []
        for ele1 in np.arange(self.epochs):

            if self.modeGradDescent == 'online':
                """Randomly shuffle the order of feeding the training data for each epoch"""
                np.random.shuffle(arr)
                self.stochastic_gradient_descent(trainData,trainDataLabels,arr)

            elif self.modeGradDescent == 'batch':
                self.batch_gradient_descent(trainData,trainDataLabels)

            print('Epoch: {0}/{1}, loss function value: {2:.1f}'.format(ele1+1, self.epochs, self.costFunctionValue))


    def predict_nn(self,testData,testDataLabels):
         # testData should be of shape numFeatures x numTestcases
        self.testDataPredictedLabels = np.zeros(testDataLabels.shape)
        self.forwardpass(testData)
        self.testDataPredictedLabels = self.predictedOutput
















