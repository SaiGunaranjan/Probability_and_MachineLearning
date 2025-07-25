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

30/08/2024
8. Numerically stable version of softmax function to prevent overflows and underflows while evaluating e^x , where x is either very large or very small.
https://medium.com/@ravish1729/analysis-of-softmax-function-ad058d6a564d

9. Successfully verified and tested the iris data set

10. Plotting both the training loss and validation loss at each epoch. A network has truly learnt
if both the training loss and the validation loss keep decreasing with epochs. If only the training
loss decreases but the validation loss increases with epochs, then the model/network has not truly
generalized on unseen data and it has probably over fit on the training data.

11. While feeding training data to a NN, all the classes/labels of the data should be in similar proportion.
If the training data is skewed towards a few classes, then the model will not learn properly and will
perfrom poorly on data from other classes which have less representation in training.

02/09/2024
12. Implemented the mini-batch mode of gradient descent

03/09/2024
13. Plotting the confusion matrix



Reference:
https://medium.com/@ja_adimi/neural-networks-101-hands-on-practice-25df515e13b0

"""

"""
NN for classification problem

1. Inititalize all the weights. How to initialize W matrices of each layer[Done. Random sampling from a uniform distribution]
2. Normalize input features to zero mean and unit variance, so that no one features totally dominates the output.[Done]
3. Add momentum term to the gradient descent algo
4. Variable learning rate/step size i.e large step size initially and smaller step size as we progress over more iterations
5. Batch vs online vs mini batch mode of gradient descent. [Done]
6. Cross validation with validation dataset (k fold cross validation)
7. How are Accuracy and loss curves computed on the validation dataset [Done]
8. Make provision for batch mode and mini batch mode of training as well.[Done]
9. Clean up multiple calls of the is else for the activation and derivative of activation function. [Done]
10. Compare my implementation of the forward pass with tensorflow/pytorch implementation [Done]
11. Check whether backward pass is correct.[Done]
12. Check if the weights update step is correct.[Done. It is correct]
13. Cost/loss function is not changing at all with epochs! [Done. I just had to increase the number of epochs from 10k to 100k for the cost function to converge and come close to 0]
14. Remove multiple passing of the 'stepSize' parameter to several functions! [Done]
15. Compute confusion matrix[Done]
16. Stop training process after training and validation accuracy both hit 90%[Done]
17. Find out failure cases[Done]
18. For batch normalization, check and verify that the gradient of the loss fn wrt to the bias is 0
(since bias gets removed by normalization step!). Verify this.

If validation loss is not changing, try changing the follwing parameters:
    1. learning rate
    2. Weight initialization
    3. Batch normalization (lamda value for convex combination)
    4. check output values for each layer
    5. Back propagated errors at each layer
    6. gradients computed
    7. Ratio of weight gradients to the original weight values

02/06/2025

Implement the batch normalization for DNN

In this commit, I have implemented the batch normalization(BN) for a DNN architecture(BN for CNN is pending, will implement it next).
The implementation is exactly along the lines of Andrej Karpathy's video lecture
https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7

In BN, we normalize ita to a standard gaussian by removing the mean and normalizing by the sigma.
Mean and sigma are computed over the mini batch. This normalized ita is then scaled by a parameter
gamma and shifted by a parameter beta. Why are we doing this?
Essentially, when training a neural network, the weight initialization is very critical,
especially for very deep networks. We have to carefully follow the He/Xavier initializations
which scale the standard normal distribution by the square root of the fan-in/number of input
connections to that layer. There is also a gain factor which depends on the type of activation function
used. This weight initialization ensures the variance across the layers stays in control and doesnt
blow up. But this kindof of very careful weight nitialization can become tedious. Instead, we adopt a
more structured approach called batch normalization. Here, at initialization, we force each layer ita
to a standard normal distribution by removing mean and scaling by sigma. We then scale back the
normalized ita with learnable parameters gamma and beta. Over iterations/epochs, the network learns
its own distribution based on gamma and beta parameters. By adoptin this approach, at initialization,
all neurons will stay active without dying or exploding. Across epochs, each neuron at each layer
will eventually learn its own distribution of values through its control parameters gamma and beta.
I have also derived the back propagation to obtain the gamma and beta gradients wrt loss function.
It seems to be working fine. I will add the derivation in my one note book. I'm able to achive decent
accuracy, though it appears slightly inferior to the He initialization. Ideally the performance has
to be same as He initialization, but I see a very slight degraded performance with BN in terms of the
loss/accuracy convergence rates. Previously, with the He initialization, it was converging faster.
Now it is taking slightly longer. I will reverify this. It could be due to some parameter setting.

While training, we continuously estimate a running mean and variance of each neuron of each layer as
a linear combination (convex combination) of the curent batch mean/sigma and the previous mean/sigma.
At the end of the training, this running mean/sigma is used while running on the test data. This way,
we don't have to explicitly compute the mean and sigma of the test data at each layer and each neuron.
We can simply resuse the running mean/sigma estimated from training data. This is just a compute
convenience. Towards this, I have introduced a flag called 'trainOrTestString' in the forward pass.
When this flag is set to 'train', a running mean/sigma is computed to normalize the ita.
When the flag is set to 'test', we use the running mean/sigma to standard normalize the ita of
test data and then scale and shift with the learnt parameters gamma and beta.

I have also added a 3rd argument in the DNN architecture definition. 1st argumenet is the number of
nodes for that layer, 2nd argument is the activation function for that layer, now 3rd argument indicates
whether BN is to be enabled or disabled for that layer. The input and output layers always should have
BN set to 0!

Next I will implement the Batch Normalization for CNN.[Done]

27/06/2025
I have tested the iris dataset with a simple ANN and batch normalization.
With the batch normalization, the training and validation accuracy have improved and it is hitting
95% training and validation accuracy. However, the test accuracy seems to be badly hit! It has
dropped to 83%. Is it the running mean and variance which is causing this issue? I need to check this.

Also, CNN has some bugs in updating the kernels weights[I believe I have now fixed the bug]

22/07/2025
1. Keep Batch normalization into a separate function instead of corrupting the forward pass




"""

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
np.random.seed(0)
from scipy.signal import convolve2d
import time as time
# import cupy as cp
# from cupyx.scipy.signal import convolve2d
# cupyx.scipy.signal.correlate2d()
# import sys

class MLFFNeuralNetwork():

    def __init__(self,networkArchitecture):

        self.networkArchitecture = networkArchitecture
        """ weights initialization between each successive layers"""
        self.numLayers = len(self.networkArchitecture)
        self.weightMatrixList = []
        self.gammaList = []
        self.betaList = []
        self.runMeanList = []
        self.runVarList = []
        self.numParamsEachDenseLayer = []
        for ele in range(self.numLayers-1):
            """Weight matrix from layer l to layer l+1 """
            numNodesLayerL = self.networkArchitecture[ele][0]
            numNodesLayerLplus1 = self.networkArchitecture[ele+1][0]
            """ Initialize the weights matrix to 0s. Other initializations like picking from a normal distribution are also possible"""
            # Initialization of weights with 0s doesnt seem to be convering for the stochastic gradient descent method!
            # weightMatrix = np.zeros((numNodesLayerLplus1,numNodesLayerL+1),dtype=np.float32) # +1 is for the bias term
            """ Initialize the weights matrix to random uniformly drawn values from 0 to 1"""
            # weightMatrix = np.random.rand(numNodesLayerLplus1,numNodesLayerL+1) # +1 is for the bias term

            # Batch Normalization (BN) not defined for input and output layers
            if (self.networkArchitecture[ele][2] == 1): # If BN is enabled for a hidden layer
                weightMatrix = np.random.randn(numNodesLayerLplus1,numNodesLayerL+1) # +1 is for the bias term
                gammaScaling = np.ones((numNodesLayerL,)) # This is the standard deviation parameter
                betaShift = np.zeros((numNodesLayerL,)) # This is the mean parameter

                """ Below 2 arrays runningMean and runningVar are not parameters for NN. So no gradients required for these"""
                runningMean = np.zeros((numNodesLayerL,))
                runningVar = np.ones((numNodesLayerL,))
            else:
                """ Weight initialization using He method"""
                fan_in = numNodesLayerL
                scalingFactorHeInit = (np.sqrt(2/fan_in)) # This is the scaling for ReLU activation functions in the DNN
                weightMatrix = np.random.randn(numNodesLayerLplus1,numNodesLayerL+1) * scalingFactorHeInit # +1 is for the bias term
                gammaScaling = np.empty([0])
                betaShift = np.empty([0])

                runningMean = np.empty([0])
                runningVar = np.empty([0])


            numParamsEachDenseLayer = weightMatrix.size + gammaScaling.size + betaShift.size
            self.numParamsEachDenseLayer.append(numParamsEachDenseLayer)
            self.weightMatrixList.append(weightMatrix)
            self.gammaList.append(gammaScaling)
            self.betaList.append(betaShift)
            self.runMeanList.append(runningMean)
            self.runVarList.append(runningVar)

        self.epsillon = 1e-8 # To take care of division by a 0 in the denominator



    def set_model_params(self,modeGradDescent = 'online',batchsize = 1, costfn = 'categorical_cross_entropy',epochs = 100000, stepsize = 0.1):
        """ By default, it is set to online/stochastic mode of GD which has a batch size = 1"""
        self.modeGradDescent = modeGradDescent
        self.costfn = costfn
        self.epochs = epochs
        self.stepsize = stepsize
        # Define batch size only for mini_batch mode of gradient descent
        if (self.modeGradDescent == 'mini_batch'):
            self.batchsize = batchsize # Batch size is typically a power of 2



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
        z = z - np.amax(z,axis=0)[None,:] # To avoid overflows while evaluating np.exp(z). Refer point 8 above
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
            mask = self.predictedOutput !=0 # Avoid 0 values in log2 evaluation. But this is not correct. It can mask wrong classifications.
            batchSize = self.predictedOutput.shape[1]
            # cost fn = -Sum(di*log(yi))/N, where di is the actual output and yi is the predicted output, N is the batch size.
            costFunction = (-np.sum((trainDataLabel[mask]*np.log2(self.predictedOutput[mask]))))/batchSize # Mean loss across data points
        elif (self.costfn == 'squared_error'):
            batchSize = self.predictedOutput.shape[1]
            # cost fn = 1/2 Sum((yi - di)**2)/N, where di is the actual output and yi is the predicted output, N is the batch size
            costFunction = (0.5*np.sum((self.predictedOutput - trainDataLabel)**2))/batchSize # Mean loss across data points. Have not verified this line. Might throw error because of shape mismatches, etc
            """ Need to divide by N (batch size) to get the mean loss across data points"""

        return costFunction


    def forwardpass(self, trainDataSample, trainOrTestString):

        """ Forward pass"""
        layerLOutput = trainDataSample
        numTrainingSamples = trainDataSample.shape[1]
        self.Ita = []
        self.outputEachlayer = []
        self.ItaNormalized = []
        self.batchMeanEachLayer = []
        self.batchVarEachLayer = []
        ## Debug
        # import pickle
        # with open("weightMatrix.pkl", "rb") as file:
        #     self.weightMatrixList = pickle.load(file)
        ##
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

                if (self.networkArchitecture[ele3][2] == 1): # If BN is enabled for a hidden layer

                    if trainOrTestString == 'train':
                        batchMean = np.mean(itaLayerL,axis=1)
                        batchVariance = np.var(itaLayerL,axis=1) # Computing variance instead of std so that division by small std can be taken care of
                        lamda = 0.9 # Should be between 0 and 1. If batch size is too small, lamda should give more weight to past mean, since current mean will keep jumping with new batches. But if batch size is very large, then we can can give more weight to current mean estimate
                        # Define running mean and running Var to be used later for testing and validation
                        self.runMeanList[ele3] = lamda * self.runMeanList[ele3] + (1-lamda)*batchMean
                        self.runVarList[ele3] = lamda * self.runVarList[ele3] + (1-lamda)*batchVariance
                    else: # For test and validation data, use the running mean and variance obtained in training
                        batchMean = self.runMeanList[ele3]
                        batchVariance = self.runVarList[ele3]

                        # batchMean = np.mean(itaLayerL,axis=1)
                        # batchVariance = np.var(itaLayerL,axis=1)


                    itaLayerLNormalized = (itaLayerL - batchMean[:,None])/np.sqrt(batchVariance[:,None] + self.epsillon)
                    gammaScaling = self.gammaList[ele3][:,None]
                    betaShift = self.betaList[ele3][:,None]
                    itaLayerL = gammaScaling*itaLayerLNormalized + betaShift



                else:
                    itaLayerLNormalized = np.empty([0])
                    batchMean = np.empty([0])
                    batchVariance = np.empty([0])

                activationFn = self.networkArchitecture[ele3][1] # Activation function name
                layerLOutput = self.activation_function(itaLayerL,activationFn) # gives output of the activation function for the ita input

                layerLminus1Output = np.ones((numNodesLayerL + 1, numTrainingSamples),dtype=np.float32) # +1 is to account for the bias term
                layerLminus1Output[1::,:] = layerLOutput

                self.Ita.append(itaLayerL) # ita is not stored for input layer. It is stored for all other layers.
                self.ItaNormalized.append(itaLayerLNormalized) # Not stored for input layer
                self.batchMeanEachLayer.append(batchMean) # Not stored for input layer
                self.batchVarEachLayer.append(batchVariance) # Not stored for input layer

            self.outputEachlayer.append(layerLminus1Output) # Output for each layer
        self.predictedOutput = layerLOutput



    def backwardpass(self, trainDataLabel):
        # This backward pass computation is for 'categorical_cross_entropy', 'squared error' loss functions

        # numTrainingSamples = trainDataLabel.shape[1]
        self.errorEachLayer = []
        # ele4 loop goes from layer L-1(output) to layer 0 input
        for ele4 in range(self.numLayers-1,0,-1):
            if (ele4 == self.numLayers-1):
                """ Final output layer""" # Does not have BN
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
                if (self.networkArchitecture[ele4][2] == 1): # If BN is enabled for a hidden layer
                    errorLayerL = errorLayerL * (self.gammaList[ele4][:,None] / np.sqrt(self.batchVarEachLayer[ele4-1][:,None] + self.epsillon))
                errorLayerLplus1 = errorLayerL

            self.errorEachLayer.append(errorLayerLplus1) # These error arrays are packed from layer L-1 down to 1 and not from 1 to L-1. They are arranged in reverse order. (layers start from 0 to L-1)
        # Errors/delta obtained for all the layers from 2 to L



    def compute_forward_backward_pass(self, trainDataSample, trainDataLabel):

        """ Forward pass"""
        self.forwardpass(trainDataSample,'train')

        """ Cost function computation"""
        self.costFunctionValue = self.compute_loss_function(trainDataLabel)

        """ Backward pass"""
        self.backwardpass(trainDataLabel)

        """ Update weights"""
        self.update_weights()


    def update_weights(self):
        # Errors/delta obtained for all the layers from 2 to L
        # Compute gradient wrt Wl_ij
        # dJ/dWl_ij = deltal+1_j * yi_l
        # gradient # dJ/dwij for all i,j
        count = -1
        for ele4 in range(self.numLayers-1):
            batchSize = self.errorEachLayer[count].shape[1]
            gradientCostFnwrtWeights = (self.errorEachLayer[count] @ self.outputEachlayer[ele4].T)/batchSize # Division becuase we want to get the mean of the gradients across all data points
            self.weightMatrixList[ele4] = self.weightMatrixList[ele4] - self.stepsize*gradientCostFnwrtWeights # Gradient descent step
            count -= 1

        # Can ideally optimize this loop and put it into above loop as well. I will do this later
        for ele5 in range(1,self.numLayers): # This starts from 1 since 0th layer i.e input layer anyways has no BN
            if (self.networkArchitecture[ele5][2] == 1):
                gradientCostFnwrtGammaScaling = np.mean(self.errorEachLayer[self.numLayers-1-ele5] * self.ItaNormalized[ele5-1], axis=1) # delta^l * ita^^l
                gradientCostFnwrtBetaShift = np.mean(self.errorEachLayer[self.numLayers-1-ele5], axis=1) # delta^l is arranged in reverse order
                self.gammaList[ele5] = self.gammaList[ele5] - self.stepsize*gradientCostFnwrtGammaScaling
                self.betaList[ele5] = self.betaList[ele5] - self.stepsize*gradientCostFnwrtBetaShift


    def train_nn(self,trainData,trainDataLabels,split = 1):
        # trainDataLabels should also be a 1 hot vector representation for classification task
        """ split tells what fraction of the data should be used for traninging and the remianingpart will be used for validation
        split (0,1]"""
        """ Split data into training and validation data. Use validation data to test model on unseeen data while training"""
        numDataPoints = trainData.shape[1]
        numTrainingData = int(np.round(split*numDataPoints))
        self.trainData = trainData[:,0:numTrainingData]
        self.trainDataLabels = trainDataLabels[:,0:numTrainingData]
        self.validationData = trainData[:,numTrainingData::]
        self.validationDataLabels = trainDataLabels[:,numTrainingData::]

        self.backpropagation()


    def stochastic_gradient_descent(self):

        numTrainData = self.trainData.shape[1]
        arr = np.arange(numTrainData)
        """Randomly shuffle the order of feeding the training data for each epoch"""
        np.random.shuffle(arr)
        """ arr is the randomly shuffled order of sampling the training data"""
        for ele2 in arr:
            trainDataSample = self.trainData[:,ele2][:,None]
            trainDataLabel = self.trainDataLabels[:,ele2][:,None]
            self.compute_forward_backward_pass(trainDataSample,trainDataLabel)

        """ Training loss and accuracy post each epoch"""
        self.compute_train_loss_acc()



    def batch_gradient_descent(self):

        trainDataSample = self.trainData
        trainDataLabel = self.trainDataLabels

        self.compute_forward_backward_pass(trainDataSample,trainDataLabel)

        """ Training loss and accuracy post each epoch"""
        self.compute_train_loss_acc()


    def mini_batch_gradient_descent(self):

        numTrainData = self.trainData.shape[1]
        arr = np.arange(numTrainData)
        """Randomly shuffle the order of feeding the training data for each epoch"""
        np.random.shuffle(arr)
        """ arr is the randomly shuffled order of sampling the training data"""
        trainDataShuffle = self.trainData[:,arr]
        trainDataLabelsShuffle = self.trainDataLabels[:,arr]
        numTrainingData = self.trainData.shape[1]
        numBatches = int(np.ceil(numTrainingData/self.batchsize))
        startIndex = 0
        for ele in range(numBatches):
            if (startIndex+self.batchsize <= numTrainingData):
                trainDataSample = trainDataShuffle[:,startIndex:startIndex+self.batchsize]
                trainDataLabel = trainDataLabelsShuffle[:,startIndex:startIndex+self.batchsize]
            else:
                trainDataSample = trainDataShuffle[:,startIndex::]
                trainDataLabel = trainDataLabelsShuffle[:,startIndex::]

            self.compute_forward_backward_pass(trainDataSample,trainDataLabel)

            startIndex += self.batchsize

        """ Training loss and accuracy post each epoch"""
        self.compute_train_loss_acc()




    def backpropagation(self):

        self.trainingLossArray = []
        self.validationLossArray = []
        for ele1 in np.arange(self.epochs):

            if self.modeGradDescent == 'online':
                self.stochastic_gradient_descent()

            elif self.modeGradDescent == 'batch':
                self.batch_gradient_descent()

            elif self.modeGradDescent == 'mini_batch':
                self.mini_batch_gradient_descent()

            if (self.validationData.shape[1] != 0): # There is some validation data to test model
                self.model_validation()

            if (self.validationData.shape[1] != 0): # There is some validation data to test model
                print('\nEpoch: {0}/{1}'.format(ele1+1, self.epochs))
                print('train_loss: {0:.1f}, val_loss: {1:.1f}, train_accuracy: {2:.1f}, val_accuracy: {3:.1f}'.format(self.trainingLoss, self.validationLoss, self.trainAccuracy, self.validationAccuracy))
                if ((self.trainAccuracy > 94) and (self.validationAccuracy > 94)): # 94
                    break
            else: # There is no validation data to test model
                print('Epoch: {0}/{1}, train_loss: {2:.1f}'.format(ele1+1, self.epochs, self.trainingLoss))


    def model_validation(self):

        self.predict_nn(self.validationData)
        """ Validation loss"""
        self.validationLoss = self.compute_loss_function(self.validationDataLabels) # Keep appending the cost function value across epochs
        self.validationLossArray.append(self.validationLoss)

        """ validation accuracy"""
        self.get_accuracy(self.validationDataLabels, self.predictedOutput)
        self.validationAccuracy = self.accuracy



    def predict_nn(self,testData):
         # testData should be of shape numFeatures x numTestcases
        self.forwardpass(testData,'test')
        self.testDataPredictedLabels = self.predictedOutput



    def get_accuracy(self, trueLabels, predLabels, printAcc=False):
        predClasses = np.argmax(predLabels,axis=0)
        actualClasses = np.argmax(trueLabels,axis=0)
        self.accuracy = np.mean(predClasses == actualClasses) * 100
        if printAcc:
            print('\nAccuracy of NN = {0:.2f} % \n'.format(self.accuracy))


    def compute_train_loss_acc(self):
        """ Compute training loss and accuracy on the training data again with the weights obtained at the end of each epoch"""
        self.forwardpass(self.trainData,'test') # Compute forward pass output on the entire training data after each epoch
        self.trainingLoss = self.compute_loss_function(self.trainDataLabels)
        self.trainingLossArray.append(self.trainingLoss) # Keep appending the cost/loss function value for each epoch
        self.get_accuracy(self.trainDataLabels, self.predictedOutput)
        self.trainAccuracy = self.accuracy


    def plot_confusion_matrix(self, trueLabels, predLabels, classLabels):

        predClasses = np.argmax(predLabels,axis=0)
        actualClasses = np.argmax(trueLabels,axis=0)
        cm = confusion_matrix(actualClasses, predClasses)
        # Plot confusion matrix
        plt.figure(figsize=(20, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=classLabels, yticklabels=classLabels)
        plt.title('Confusion Matrix. Test accuracy = {0:.2f} %'.format(self.accuracy))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()


""" Convolutional Neural Networks

Need to go from ANN to CNN?
https://www.quora.com/Why-do-we-use-CNN-when-we-already-have-ANN-with-a-fully-connected-structure
https://www.youtube.com/watch?v=H_JtjeSNhA0&list=PLEAYkSg4uSQ0Q5Z1IYI-0g2cbD-2Rt-I6&index=33
1. Exploding number of ANN parameters/weights especially when the size of input image is very large and we have to flatten for the ANN
2. Larger the input size, more the number of weights/parameters --> more number of examples
required to train the network.
3. For an ANN, an image with cat at top left corner of image is different from an image with cat at
bottom right of the image, so it treats it as two different outputs. Whereas, a CNN does a
local weighting/activation and hence for a CNN, both the images are treated the same.
Hence we will move to CNNS for image datasets.

A typical CNN consists of convolutional layers, pooling layers, flattening of the final convolutional layer
followed by a typical ANN layers. For the ANN part, I have reused the MLFFNeuralNetwork class.
This script, contains both the ANN and CNN class.
Since the CNN has convolutions (multi dimensional), the compute is on the higher side as compared to
a simple ANN.


Updates:

23/09/2024
1. Started the initial understanding and coding of the CNN and backpropagation

02/10/2024
1. Implemented Convolutional neural network from scratch

In this script, I have implemented the Convolutional Neural Network (CNN) from scratch!
This includes the backpropagation algorithm for CNNs. I have derived the backpropagation algorithm
for the CNNs on papaer all by myself and then implemented the same in the code.
The derivation is based on my understanding of the backprop for ANN from Shastry's lectures.
I was able to extend the same for CNN backprop. And it worked in the first shot!
You do not find the complete derivation of the backprop for CNNs anywhere. Most of the blogs or
video tutorials just show for 1 layer with 1 channel. But, I was able to derive it completely!
I tested the CNN on the MNIST digit database and was able to achieve a training accuracy of 93% !
I have implemented all the flavours of Gradient descent like, stochastic gradient descent, batch,
mini-batch.

2. I will add the derivation of the back prop for CNNs into my one note as well

3. The motivation for CNN backprop was from Dr. Vineeth Balasubramaniam's  course on CNNs.
Link:
    https://www.youtube.com/watch?v=pUCCd2-17vI&list=PLEAYkSg4uSQ0Q5Z1IYI-0g2cbD-2Rt-I6&index=33


Good reference docs for GPU parallelization:
    https://numba.pydata.org/numba-doc/0.13/CUDAJit.html

    https://numba.readthedocs.io/en/stable/cuda/index.html

26/05/2025
Achieved 85% training and validation accuracy for the Fashion MNIST dataset!

Finally after nearly 2 months of struggle, I have got a decent accuracy for the fashion MNIST dataset with the CNN! The issue was with the weight initialization! My back propagation is absolutely correct! (previously, I was doubting my backporpagation implementation!) Now, I'm able to achieve about 85% train and validation accuracy with a few epochs! The following the changes which helped in achieveing this.

He initialization instead of normal distribution initialization
He initialization is basically scaling the normal distribution by the sqrt of the nunber of inputs to that layer and sometimes also a multificative gain factor. This gain factor is a function of the activation function used. For example, ReLU has sqrt(2). tanh has 5/3 and so on. Currently, I have scaled based on ReLU only! Need to extend it to other activations as well.

Changed the loss and gradient computation from sum of all datapoints to mean of all datapoints
The loss now is the mean loss of all data points and the gradient is the mean of the gradients of all data points. Now we see reasonable loss values and also the netwrok starts respoding due to more reasonable gradeint values (with sum, gradients blow up)

Implemented the prediction/test also as a batch mode. Now the testing time also is reduced! Previously only the training was parallelized and testing was given as a bulk. Due to this the system was getting stuck or crashing. Now I have parallelized the testing part as well.

Next I will implement the batch normalization to further improve the accuracy

Action Items

1. Ensure size of kernel at any stage is smaller than size of input. Handle this gracefully, else it might crash [Done]
2. All pool layers have to be same type. Either all have to be maxpool or all have to be avg pool(for now). I'm not handling a mix of max and avg pools in this script
3. Rename outputEachConvlayer to outputEachConvlayerPostPool
4. Not clear how to perform derivative of activation function (f'(ita)) for avg pooling. Especially for activation functions other than ReLU [Clear now!]
5. Multiple definitions of numTrainData in mini_batch_gradient_desc and mini_batch_gradient_desc_cnn
6. Change the trainAccuracy and ValidationAccuracy exit condition to 95%
7. Make step size smaller and smaller as the training and validation accuracy goes beyond 90% and you wish to achieve a better accuracy
8. Save model and run without training
9. Support average pooling as well
10. Understand dropout layer[Understood, but need to implement]
11. Understand batch normalization[Done. Implemented as well]
12. Print the number of parameters in the model[Done]
13. Depthwise convolution + point wise convolution has lesser number of parameter and reduces
number of operations as compared to standard convolution. But how do we learn the parameters/backprop
in depth-wise and point wise convolution
14. Threads per block, blocks per grid are not correctly optimized. This needs tweaking to get optimal GPU utilization/performance
15. Parallelization across data is much faster than parallelization across kernels especially when number of kernels in each layer is small like MNIST CNN dataset
16. Parallelize maxpooling.[Done]
17. backprop_poollayer/reverse pooling also has to be converted to a kernel. Eventually make everything on GPU[Done]
18. Parallelization across kernels not completely tested[Done]
19. When data is large, use parallelization across data and if data is small then use parallelization across kernels
20. Print size of memory moved to GPU kernel[Done]
21. For computing the accuracy and cost function for the training data (post the training) after each epoch,
the size of the bulk training data will be huge, need to compute in chunks and not as a bulk,
else GPU memory will be insufficient[Done]
22. For the fashion MNIST dataset, the loss function is not changing at all meaning the weights are not getting updated at all. [Done]
Debugged this. Cause was the outputs were too large especially at the output layer logits before taking the softmax. This implied that
internally, the outputs at each layer were blowing up. Now, since the logits were too large, post soft max, the output values were always zero or 1 and not any intermediate values!
Due to this, when we were computing the error, it was coming out as 0 even at the first iteration(even with random weights)!
Why? Because, if the output was correct, error is 0 and error propagated is also 0. Also, I was calculating the loss and gradients as a sum for all data points.
This was blowing up the loss  and the gradients as well leading to exploding gradients as well. Now I have replaced sum with mean and now compute the loss and gradeients
as a mean of all datapoints.
This means the gradient also becomes zero, vanishing gradients(Note: Gradient is product of error at next layer and output of current layer)
Since gradients were zero, the weights were also not updating and hence no change in loss. I have fixed this by using the He method of weight initialization!
23. Loss fn isn't changing means either the update rate is too small or that the gradients are too large or too small. For the fashion MNIST dataset, it was
vanishing gradients. [Done] Fixed by He weight initialization!
24. Implement RMSProp, ADAM, Adagrad methods of Gradient descent optimization algorithms
25. Xavier and He initialization methods. Video link: https://www.youtube.com/watch?v=1pgahpCZeF0&list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT&index=73
26. Dropouts for regularization
27. Batch normalization [Done]
28. Plot distribution of weights at each layer
29. If we are going with a simple optimizer like gradient descent,
we should follow the standard initialization practises like He, Xavier which ensure all neuroans are alive
and learning throughout the training process. But modern techniques like dropouts, normalizations, other optimizers
like RMSProp, Adam, we dont need to worry too much about initializations. These methods handle relatively poor initializations as well.
The below lecture link by Andrez Karpathy beautifully explains this:
    https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4
There is also another lecture by Mitesh Khapra, IITM, which talks about the initilaizations in a more detailed manner.
30. When using batch normalization, we are making the ita as normal gaussian distribution and hence it has zero mean which means the bias term that we add
earlier is getting removed post the batch normalization. Infact if we check the gradients of the bias terms with batch normalization, they will all be 0.
So, if we are enabling batch normalization, we can get rid of the bias terms! As an exercise, after implementing batch normalization, check the gradients of the bias term!
The bias term we add in the batch normalization process, itself takes care of the bias
31. If a layer has batch normalization, then replace f'(ita) with f'(ita_^^)*(alpha/sigma)
where ita_^^ = alpha (ita_^) + beta, ita_^ = (ita-mu)/sigma, where mu and sigma are the mean and variance across the data points for that particular node/neuron.
I have derived this, but need to verify! [Done]
31. Understand the embedding matrix for LLMs. Why is it used?
32. Plot running mean and variance across batches
33. Numrical gradient check?
34. Regularization and dropouts, adam optimizer
35. check if gradient wrt bias term is 0 with BN enabled
36. What shoud be the scaling factor for the l2 regularization
37. Is BN implemented correctly in CNN? [Yes! Im now getting better results with BN on fashion MNIST dataset with a simpler CNN architecture]


Batch normalization is used only for hidden layers. It should not be used for the final output
layer before softmax activation!


With the below CNN architecture on the fashion MINIST dataset,
 Reference architecture from:
    https://medium.com/@sanjay_dutta/building-a-baseline-convolutional-neural-network-for-fashion-mnist-600634e5feef

The following are the observations:
1. Without BN for both CNN and DNN, train accuracy = 89%, validation accuracy = 89%, test accuracy = 89%
2. With BN for both CNN and DNN, train accuracy = 86%, validation accuracy = 86%, test accuracy = 86%. But I was expecting much better results with BN enabled! Not sure why this is happening. Need to debug the reason for this. May need to add regularization and/or drop outs.
3. With BN for DNN and without BN for CNN, train accuracy = 85%, validation accuracy = 85%, test accuracy = 85%. This means that BN in DNN itself is limiting the perforamance.
4. With BN for CNN and without BN for DNN, train accuracy = 94%, validation accuracy = 91%, test accuracy = 91%. This implies that BN for DNN is limiting the performance and that the CNN code and BN for CNN has been implemented correctly!


"""

from cnn_gpu_kernels.parallelize_across_datapoints import cnn_convolve2d_gpu, cnn_backward_convolve2d_gpu, \
    cnn_gradient_convolve2d_gpu, pooling_gpu, backprop_poollayer_gpu

from cnn_gpu_kernels.parallelize_across_kernels import cnn_convolve2d_parallel_ker_gpu, cnn_backward_convolve2d_parallel_chan_gpu, \
    cnn_gradient_convolve2d_parallel_ker_gpu, pooling_gpu_parallel_ker, backprop_poollayer_gpu_parallel_ker

class ConvolutionalNeuralNetwork():

    def __init__(self, inputShape, convLayer, poolLayer, denseLayer, outputLayer):

        self.inputShape = inputShape # Numchannles, l, w
        #(#filters, size of kernel(length), activation function)
        self.convLayer = convLayer # Len of this list is the number of convolutional layers
        self.poolLayer = poolLayer # size,stride, type of pool. There will be a pooling layer for every convolutional layer.
        #(#nodes, activation function)
        self.denseLayer = denseLayer # Len of this list indicates number of hidden layers
        self.outputLayer = outputLayer

        """ Convolutional kernels and pool layers are always assumed to be having same height and width i.e they are square shaped"""
        # networkArchitecture = [inputShape, convLayer, denseLayer, outputLayer]
        """ Kernel weights initialization for convolutional layers"""
        inputDepth = self.inputShape[0]
        inputHeight, inputWid = self.inputShape[1], self.inputShape[2]
        self.kernelWeights = []
        self.bias = []
        self.numConvLayers = len(self.convLayer)
        self.numParamsEachConvLayer = []
        self.gammaList = []
        self.betaList = []
        self.runMeanList = []
        self.runVarList = []

        for ele in range(self.numConvLayers):
            """ Conv layer"""
            numFilters = self.convLayer[ele][0]
            filterSize = self.convLayer[ele][1] # FilterSize/KernelSize

            if (self.convLayer[ele][3] == 1): # If BN is enabled for a hidden layer
                kernelWeights = np.random.randn(numFilters,filterSize,filterSize,inputDepth) # +1 is for the bias term
                bias = np.random.randn(numFilters)
            else:
                """ Weight initialization using He method"""
                fan_in = filterSize * filterSize * inputDepth
                scalingFactorHeInit = np.sqrt(2/fan_in) # For ReLU activation function only! This changes based on the activation function used
                kernelWeights = np.random.randn(numFilters,filterSize,filterSize,inputDepth) * scalingFactorHeInit
                bias = np.random.randn(numFilters) * scalingFactorHeInit
            # Below is the size/dimensions of the output post convolution
            inputDepth = numFilters
            inputHeight, inputWid = inputHeight-filterSize+1, inputWid-filterSize+1 # Post "valid" convolution

            if (self.convLayer[ele][3] == 1):
                gammaScaling = np.ones((inputDepth,inputHeight,inputWid)) # This is the standard deviation parameter
                betaShift = np.zeros((inputDepth,inputHeight,inputWid)) # This is the mean parameter
                """ Below 2 arrays runningMean and runningVar are not parameters for NN. So no gradients required for these"""
                runningMean = np.zeros((inputDepth,inputHeight,inputWid))
                runningVar = np.ones((inputDepth,inputHeight,inputWid))
            else:
                gammaScaling = np.empty([0])
                betaShift = np.empty([0])
                runningMean = np.empty([0])
                runningVar = np.empty([0])

            numParams = kernelWeights.size + bias.size + gammaScaling.size + betaShift.size
            self.numParamsEachConvLayer.append(numParams)
            self.kernelWeights.append(kernelWeights)
            self.bias.append(bias)
            self.gammaList.append(gammaScaling)
            self.betaList.append(betaShift)
            self.runMeanList.append(runningMean)
            self.runVarList.append(runningVar)

            """ Pooling layer"""
            poolSize = self.poolLayer[ele][0]
            poolStride = self.poolLayer[ele][1]
            # Below is the size/dimensions of the output post pooling
            """ Verify below formula once"""
            inputHeight, inputWid = (inputHeight-poolSize)//poolStride + 1, (inputWid-poolSize)//poolStride + 1

        numNodesPostFlatten = (inputHeight * inputWid * inputDepth)#.astype(np.int32)
        flattenLayer = [(numNodesPostFlatten,'Identity',0)] # input to dense layer will not have BN
        denseLayerArchitecture = flattenLayer + self.denseLayer + self.outputLayer
        # self.numDenseLayers = len(self.denseLayerArchitecture)
        """ Dense layer weights initialization"""
        self.mlffnn = MLFFNeuralNetwork(denseLayerArchitecture)

        self.ParamsAllLayers = self.numParamsEachConvLayer + self.mlffnn.numParamsEachDenseLayer # Append the lists of learnable parameters of convolution and dense layers
        self.totalParamsAllLayers = sum(self.ParamsAllLayers)
        print('\n Total trainable params: {0} ({1:.2f} KB) \n'.format(self.totalParamsAllLayers,(self.totalParamsAllLayers*4)/1024))
        """ Flag to run CNN on CPU or GPU"""
        self.runCNNCPU = False # True to run on CPU and False to run on GPU
        if (self.runCNNCPU == False):
            self.parallelizeAcrossData = True#True # True for GPU parallelization across data and false for parallelization across channels/kernels
            # Always try to run parallelization across data first. Time taken by parallelization across kernels is much larger
            # If batch size is much larger than number of channels/kernels, then use parallelize across data, else if number of kernels is larger
            # than batch size like in stochastic gradient descent, use parallelize across kernels

        self.epsillon = 1e-8 # To take care of division by a 0 in the denominator


    def forwardpass_cnn(self, trainDataImage, trainOrTestString):
        """ Check indices of ele3 for forward pass"""
        """ Forward pass CNN"""
        layerLOutput = trainDataImage
        numTrainingSamples = trainDataImage.shape[3] # 1st dim is numChannels, 2nd and 3rd dim are height, width, 4th dim is number of such images
        self.ItaConv = []
        self.ItaConvNormalized = []
        self.outputEachConvlayer = []
        self.maxPoolingIndexEachConvLayer = []
        self.batchMeanEachLayer = []
        self.batchVarEachLayer = []

        """Input layer"""
        # ele3 = 0, # Input/1st layer is taken as a dummy convolution layer but with no convolution
        layerLminus1Output = layerLOutput
        self.maxPoolingIndex = np.zeros(layerLOutput.shape,dtype=np.int32) # No pooling required for first output layer which is actually input layer
        self.outputEachConvlayer.append(layerLminus1Output) # Output for each layer. Stored post pooling
        self.maxPoolingIndexEachConvLayer.append(self.maxPoolingIndex)

        ## For debug only
        # import pickle
        # with open("kernelWeights.pkl", "rb") as file:
        #     self.kernelWeights = pickle.load(file)

        # with open("bias.pkl", "rb") as file:
        #     self.bias = pickle.load(file)
        ##

        # ele3 is looping over the convolution layers
        for ele3 in range(1,self.numConvLayers+1):
            weightMatrixLayerLminus1toL = self.kernelWeights[ele3-1]
            """ Convolution followed by pooling"""
            """ Currently written only for valid mode of convolution"""
            t1 = time.time()
            if (self.runCNNCPU == True):
                itaLayerL = self.cnn_convolve2d(layerLminus1Output, weightMatrixLayerLminus1toL,convMode='valid')
            else:
                if (self.parallelizeAcrossData == True):
                    itaLayerL = cnn_convolve2d_gpu(layerLminus1Output, weightMatrixLayerLminus1toL,convMode='valid')
                else:
                    itaLayerL = cnn_convolve2d_parallel_ker_gpu(layerLminus1Output, weightMatrixLayerLminus1toL,convMode='valid')
            t2 = time.time()
            # print('\n\tTime taken for forward pass convolution layer {0} is {1:.2f} ms'.format(ele3, (t2-t1)*1000))
            itaLayerL += self.bias[ele3-1][:,None,None,None]

            if (self.convLayer[ele3-1][3] == 1): # If BN is enabled for a hidden layer

                if trainOrTestString == 'train':
                    batchMean = np.mean(itaLayerL,axis=3)
                    batchVariance = np.var(itaLayerL,axis=3) # Computing variance instead of std so that division by small std can be taken care of
                    lamda = 0.9 # Should be between 0 and 1. If batch size is too small, lamda should give more weight to past mean, since current mean will keep jumping with new batches. But if batch size is very large, then we can can give more weight to current mean estimate
                    # Define running mean and running Var to be used later for testing and validation
                    self.runMeanList[ele3-1] = lamda * self.runMeanList[ele3-1] + (1-lamda)*batchMean
                    self.runVarList[ele3-1] = lamda * self.runVarList[ele3-1] + (1-lamda)*batchVariance
                else: # For test and validation data, use the running mean and variance obtained in training
                    batchMean = self.runMeanList[ele3-1]
                    batchVariance = self.runVarList[ele3-1]


                itaLayerLNormalized = (itaLayerL - batchMean[:,:,:,None])/np.sqrt(batchVariance[:,:,:,None] + self.epsillon)
                gammaScaling = self.gammaList[ele3-1][:,:,:,None]
                betaShift = self.betaList[ele3-1][:,:,:,None]
                itaLayerL = gammaScaling*itaLayerLNormalized + betaShift

            else:
                itaLayerLNormalized = np.empty([0])
                batchMean = np.empty([0])
                batchVariance = np.empty([0])

            activationFn = self.convLayer[ele3-1][2] # Activation function name
            layerLOutput = self.mlffnn.activation_function(itaLayerL,activationFn) # gives output of the activation function for the ita input
            t5 = time.time()
            """ Pooling"""
            poolLayer = self.poolLayer[ele3-1]
            if (self.runCNNCPU == True):
                layerLminus1Output = self.pooling(layerLOutput,poolLayer)
            else:
                if (self.parallelizeAcrossData == True):
                    layerLminus1Output, self.maxPoolingIndex = pooling_gpu(layerLOutput,poolLayer)
                else:
                    layerLminus1Output, self.maxPoolingIndex = pooling_gpu_parallel_ker(layerLOutput,poolLayer)
            t6 = time.time()
            # print('\n\tTime taken for forward pass Pooling layer {0} is {1:.2f} ms'.format(ele3, (t6-t5)*1000))
            self.ItaConv.append(itaLayerL) # ita is not stored for input layer. It is stored for all other layers.
            self.ItaConvNormalized.append(itaLayerLNormalized)
            self.batchMeanEachLayer.append(batchMean) # Not stored for input layer
            self.batchVarEachLayer.append(batchVariance) # Not stored for input layer
            self.outputEachConvlayer.append(layerLminus1Output) # Output for each layer. Stored post pooling
            self.maxPoolingIndexEachConvLayer.append(self.maxPoolingIndex)



        numChannelsLastConvLayer = layerLminus1Output.shape[0]
        heightLastConvLayer = layerLminus1Output.shape[1]
        widthLastConvLayer = layerLminus1Output.shape[2]
        layerLminus1Output2d = np.transpose(layerLminus1Output,(3,0,1,2)).reshape(numTrainingSamples,
                                                            numChannelsLastConvLayer*heightLastConvLayer*widthLastConvLayer)

        # According to tensorflow, first flatten channels then width and then height. This is counter intuitive!
        # layerLminus1Output2d = np.transpose(layerLminus1Output,(3,1,2,0)).reshape(numTrainingSamples,
        #                                                     numChannelsLastConvLayer*heightLastConvLayer*widthLastConvLayer)


        flattenOutputConvLayers = layerLminus1Output2d.T # For batch/mini batch mode, we will have to make it 2 D and not flatten as 1D
        self.mlffnn.forwardpass(flattenOutputConvLayers, trainOrTestString)
        self.predictedOutputcnn = self.mlffnn.predictedOutput



    def backwardpass_cnn(self,trainDataLabel):

        """ Dense layer backward pass"""
        self.mlffnn.backwardpass(trainDataLabel)

        """ Back propagating error from dense layer to last convolutional layer"""
        errorDenseLayer2 = self.mlffnn.errorEachLayer[-1]
        weightMatrixLayer1to2 = self.mlffnn.weightMatrixList[0]
        errorDenseLayer1 = (weightMatrixLayer1to2[:,1::].T @ errorDenseLayer2) # numNodesFlattenLayer x numDataPoints
        errorDenseLayer1 = np.transpose(errorDenseLayer1,(1,0)) # numDataPoints x numNodesFlattenLayer
        numDataPoints = errorDenseLayer1.shape[0]
        numChannels, height, width, _ = self.outputEachConvlayer[-1].shape # _ and numDataPointsshould be same
        errorLastConvLayerPostPool = errorDenseLayer1.reshape(numDataPoints, numChannels, height, width)
        errorLastConvLayerPostPool = np.transpose(errorLastConvLayerPostPool, (1,2,3,0))

        """Convolutional layer back propogation """
        shapeLayerLPrePooling = self.ItaConv[-1].shape
        poolProperties = self.poolLayer[-1]
        poolInds = self.maxPoolingIndexEachConvLayer[-1]
        if (self.runCNNCPU == True):
            errorLastConvLayerPrePool = self.backprop_poollayer(errorLastConvLayerPostPool, poolInds, poolProperties, shapeLayerLPrePooling)
        else:
            if (self.parallelizeAcrossData == True):
                errorLastConvLayerPrePool = backprop_poollayer_gpu(errorLastConvLayerPostPool, poolInds, poolProperties, shapeLayerLPrePooling)
            else:
                errorLastConvLayerPrePool = backprop_poollayer_gpu_parallel_ker(errorLastConvLayerPostPool, poolInds, poolProperties, shapeLayerLPrePooling)



        """ First backpropagate error from pooling layer and then multiply with derivative of activation function"""
        itaLayerL = self.ItaConv[-1]
        activationFn = self.convLayer[-1][2] # Activation fn of last convolutional layer
        activationFnDerivative = self.mlffnn.derivative_activation_function(itaLayerL,activationFn)
        errorLayerLplus1 = errorLastConvLayerPrePool * activationFnDerivative
        if (self.convLayer[-1][3] == 1): # If BN is enabled for last conv layer
            errorLayerLplus1 = errorLayerLplus1 * (self.gammaList[-1][:,:,:,None] / np.sqrt(self.batchVarEachLayer[-1][:,:,:,None] + self.epsillon))

        self.errorEachConvLayer = []
        self.errorEachConvLayer.append(errorLayerLplus1) # Appending the error of last conv layer
        # ele4 loop goes from layer L-1(output) to layer 0 input
        for ele4 in range(self.numConvLayers-1,0,-1):
            kernelWeightsLayerL = self.kernelWeights[ele4]
            t7 = time.time()
            """ Below convolution should be full correlation"""
            kernelWeightsLayerLFlipHeightWidth = np.flip(kernelWeightsLayerL,axis=(1,2))
            if (self.runCNNCPU == True):
                errorLayerL = self.cnn_backward_convolve2d(errorLayerLplus1, kernelWeightsLayerLFlipHeightWidth, convMode='full')
            else:
                if (self.parallelizeAcrossData == True):
                    errorLayerL = cnn_backward_convolve2d_gpu(errorLayerLplus1, kernelWeightsLayerLFlipHeightWidth, convMode='full')
                else:
                    errorLayerL = cnn_backward_convolve2d_parallel_chan_gpu(errorLayerLplus1, kernelWeightsLayerLFlipHeightWidth, convMode='full')
            t8 = time.time()
            # print('\n\tTime taken for backward pass convolution layer {0} is {1:.2f} ms'.format(ele4, (t8-t7)*1000))
            itaLayerL = self.ItaConv[ele4-1]
            activationFn = self.convLayer[ele4-1][2]
            activationFnDerivative = self.mlffnn.derivative_activation_function(itaLayerL,activationFn)
            """ First backpropagate error from pooling layer and then multiply with derivative of activation function"""
            shapeLayerLPrePooling = itaLayerL.shape
            poolProperties = self.poolLayer[ele4-1]
            poolInds = self.maxPoolingIndexEachConvLayer[ele4]#self.maxPoolingIndexEachConvLayer[ele4-1]
            t9 = time.time()
            if (self.runCNNCPU == True):
                errorLayerLPrePool = self.backprop_poollayer(errorLayerL, poolInds, poolProperties, shapeLayerLPrePooling)
            else:
                if (self.parallelizeAcrossData == True):
                    errorLayerLPrePool = backprop_poollayer_gpu(errorLayerL, poolInds, poolProperties, shapeLayerLPrePooling)
                else:
                    errorLayerLPrePool = backprop_poollayer_gpu_parallel_ker(errorLayerL, poolInds, poolProperties, shapeLayerLPrePooling)
            t10 = time.time()
            # print('\n\tTime taken for backward pass Pooling layer {0} is {1:.2f} ms'.format(ele4, (t10-t9)*1000))
            errorLayerLplus1 = errorLayerLPrePool * activationFnDerivative
            if (self.convLayer[ele4-1][3] == 1): # If BN is enabled for last conv layer
                errorLayerLplus1 = errorLayerLplus1 * (self.gammaList[ele4-1][:,:,:,None] / np.sqrt(self.batchVarEachLayer[ele4-1][:,:,:,None] + self.epsillon))


            self.errorEachConvLayer.append(errorLayerLplus1) # These error arrays are packed from layer L-1 down to 1 and not from 1 to L-1. They are arranged in reverse order. (layers start from 0 to L-1)
            # Errors/delta obtained for all the layers from 2 to L



    def compute_forward_backward_pass_cnn(self, trainDataSample, trainDataLabel):

        """ Forward pass"""
        t1 = time.time()
        self.forwardpass_cnn(trainDataSample, 'train')
        t2 = time.time()
        # print('Time taken for forward pass = {0:.2f} s'.format(t2-t1))

        """ Cost function computation"""
        t3 = time.time()
        self.costFunctionValue = self.mlffnn.compute_loss_function(trainDataLabel)
        t4 = time.time()
        # print('Time taken for cost fn eval = {0:.2f} s'.format(t4-t3))

        """ Backward pass"""
        t5 = time.time()
        self.backwardpass_cnn(trainDataLabel)
        t6 = time.time()
        # print('Time taken for backward pass = {0:.2f} s'.format(t6-t5))

        """ Update weights"""
        t7 = time.time()
        self.update_weights_cnn()
        t8 = time.time()
        # print('Time taken for Weight update = {0:.2f} s'.format(t8-t7))




    def update_weights_cnn(self):
        # Errors/delta obtained for all the layers from 2 to L
        # Compute gradient wrt Wl_ij
        # dJ/dWl_ij = deltal+1_j * flip(yi_l)
        # gradient # dJ/dwij for all i,j

        count = -1
        for ele4 in range(self.numConvLayers):
            if (self.runCNNCPU == True):
                gradientCostFnwrtKernelWeights = self.cnn_gradient_convolve2d(np.flip(self.outputEachConvlayer[ele4],axis=(1,2)), self.errorEachConvLayer[count], convMode='valid')
            else:
                if (self.parallelizeAcrossData == True):
                    gradientCostFnwrtKernelWeights = cnn_gradient_convolve2d_gpu(np.flip(self.outputEachConvlayer[ele4],axis=(1,2)), self.errorEachConvLayer[count], convMode='valid')
                else:
                    gradientCostFnwrtKernelWeights = cnn_gradient_convolve2d_parallel_ker_gpu(np.flip(self.outputEachConvlayer[ele4],axis=(1,2)), self.errorEachConvLayer[count], convMode='valid')

            gradientCostFnwrtKernelWeightsAllDataPoints = np.mean(gradientCostFnwrtKernelWeights,axis=-1)#np.sum(gradientCostFnwrtKernelWeights,axis=-1)
            self.kernelWeights[ele4] = self.kernelWeights[ele4] - self.mlffnn.stepsize*gradientCostFnwrtKernelWeightsAllDataPoints # Gradient descent step
            gradientCostFnwrtBias = np.mean(self.errorEachConvLayer[count],axis=(1,2,3)) #np.sum(self.errorEachConvLayer[count],axis=(1,2,3))
            self.bias[ele4] = self.bias[ele4] - self.mlffnn.stepsize*gradientCostFnwrtBias # Gradient descent step

            if (self.convLayer[ele4][3] == 1): # If BN is enabled, update gamma and beta
                gradientCostFnwrtGammaScaling = np.mean(self.errorEachConvLayer[count] * self.ItaConvNormalized[ele4], axis=-1) # delta^l * ita^^l
                gradientCostFnwrtBetaShift = np.mean(self.errorEachConvLayer[count], axis=-1) # delta^l is arranged in reverse order
                self.gammaList[ele4] = self.gammaList[ele4] - self.mlffnn.stepsize*gradientCostFnwrtGammaScaling
                self.betaList[ele4] = self.betaList[ele4] - self.mlffnn.stepsize*gradientCostFnwrtBetaShift

            count -= 1

        self.mlffnn.update_weights()
        # print('gradientCostFnwrtKernelWeightsAllDataPoints', gradientCostFnwrtKernelWeightsAllDataPoints[10,1,2,25])
        # print('kernel weight [2]', self.kernelWeights[2][10,1,2,25])




    def train_cnn(self,trainData,trainDataLabels,split = 1):
        # trainDataLabels should also be a 1 hot vector representation for classification task
        """ split tells what fraction of the data should be used for traninging and the remianingpart will be used for validation
        split (0,1]"""
        """ Split data into training and validation data. Use validation data to test model on unseeen data while training"""
        numDataPoints = trainData.shape[3]
        numTrainingData = int(np.round(split*numDataPoints))
        self.trainData = trainData[:,:,:,0:numTrainingData]
        self.trainDataLabels = trainDataLabels[:,0:numTrainingData]
        self.validationData = trainData[:,:,:,numTrainingData::]
        self.validationDataLabels = trainDataLabels[:,numTrainingData::]

        self.backpropagation_cnn()



    def stochastic_gradient_descent_cnn(self):

        numTrainData = self.trainData.shape[3]
        arr = np.arange(numTrainData)
        """Randomly shuffle the order of feeding the training data for each epoch"""
        np.random.shuffle(arr)
        """ arr is the randomly shuffled order of sampling the training data"""
        count = 1
        for ele2 in arr:
            trainDataSample = self.trainData[:,:,:,ele2][:,:,:,None]
            trainDataLabel = self.trainDataLabels[:,ele2][:,None]
            t1 = time.time()
            self.compute_forward_backward_pass_cnn(trainDataSample,trainDataLabel)
            t2 = time.time()
            # print('Training example {0}/{1}. Time taken = {2:.2f} ms'.format(count,numTrainData, (t2-t1)*1000))
            count += 1

        # print('Im here')
        """ Training loss and accuracy post each epoch"""
        t3 = time.time()
        self.compute_train_loss_acc_cnn()
        t4 = time.time()
        # print('Time taken for computing training loss and accuracy after epoch = {0:.2f} min'.format(t4-t3)/60)


    def batch_gradient_descent_cnn(self):

        trainDataSample = self.trainData
        trainDataLabel = self.trainDataLabels

        self.compute_forward_backward_pass_cnn(trainDataSample,trainDataLabel)

        """ Training loss and accuracy post each epoch"""
        t3 = time.time()
        self.compute_train_loss_acc_cnn()
        t4 = time.time()
        # print('Time taken for computing training loss and accuracy after epoch = {0:.2f} min'.format(t4-t3)/60)


    def mini_batch_gradient_descent_cnn(self):

        numTrainData = self.trainData.shape[3]
        arr = np.arange(numTrainData)
        """Randomly shuffle the order of feeding the training data for each epoch"""
        np.random.shuffle(arr)
        """ arr is the randomly shuffled order of sampling the training data"""
        trainDataShuffle = self.trainData[:,:,:,arr]
        trainDataLabelsShuffle = self.trainDataLabels[:,arr]
        numTrainingData = self.trainData.shape[3]
        numBatches = int(np.ceil(numTrainingData/self.mlffnn.batchsize))
        startIndex = 0
        for ele in range(numBatches):
            if (startIndex+self.mlffnn.batchsize <= numTrainingData):
                trainDataSample = trainDataShuffle[:,:,:,startIndex:startIndex+self.mlffnn.batchsize]
                trainDataLabel = trainDataLabelsShuffle[:,startIndex:startIndex+self.mlffnn.batchsize]
            else:
                trainDataSample = trainDataShuffle[:,:,:,startIndex::]
                trainDataLabel = trainDataLabelsShuffle[:,startIndex::]
            t1 = time.time()
            self.compute_forward_backward_pass_cnn(trainDataSample,trainDataLabel)
            t2 = time.time()
            # print('Time taken for batch {0}/{1} is {2:.2f} s'.format(ele+1,numBatches,(t2-t1)))

            startIndex += self.mlffnn.batchsize

        """ Training loss and accuracy post each epoch"""
        t3 = time.time()
        self.compute_train_loss_acc_cnn()
        t4 = time.time()
        # print('Time taken for computing training loss and accuracy after epoch = {0:.2f} min'.format(t4-t3)/60)



    def model_validation_cnn(self):

        self.predict_cnn(self.validationData)
        """ Validation loss"""
        self.validationLoss = self.mlffnn.compute_loss_function(self.validationDataLabels) # Keep appending the cost function value across epochs
        self.validationLossArray.append(self.validationLoss)

        """ validation accuracy"""
        self.mlffnn.get_accuracy(self.validationDataLabels, self.predictedOutputcnn)
        self.validationAccuracy = self.mlffnn.accuracy



    def predict_cnn(self,testData):
         # testData should be of shape numFeatures x numTestcases

        numvalidationData = testData.shape[3]
        if (self.runCNNCPU == True):
            self.forwardpass_cnn(testData,'test')
            self.testDataPredictedLabels = self.predictedOutputcnn
        else:
            if (self.parallelizeAcrossData == True):
                numBatches = int(np.ceil(numvalidationData/self.mlffnn.batchsize))
                batchsize = self.mlffnn.batchsize
            else:
                numBatches = numvalidationData
                batchsize = 1

            startIndex = 0
            numOutputNodes = self.outputLayer[0][0]
            predictedOutputAllTestData = np.zeros((numOutputNodes,numvalidationData),dtype=np.float32)
            for ele in range(numBatches):
                if (startIndex+batchsize <= numvalidationData):
                    testDataSample = testData[:,:,:,startIndex:startIndex+batchsize]
                else:
                    testDataSample = testData[:,:,:,startIndex::]

                self.forwardpass_cnn(testDataSample,'test')

                if (startIndex+batchsize <= numvalidationData):
                    predictedOutputAllTestData[:,startIndex:startIndex+batchsize] = self.predictedOutputcnn
                else:
                    predictedOutputAllTestData[:,startIndex::] = self.predictedOutputcnn

                startIndex += batchsize

            self.testDataPredictedLabels = predictedOutputAllTestData # This variable is used outside this class and in the script where the cnn class is called
            self.mlffnn.predictedOutput = predictedOutputAllTestData # This variable is populated to cater to cost function/loss function evaluation in the MLFF class method
            self.predictedOutputcnn = predictedOutputAllTestData # This variable is required for evaluating the validation accuracy




    def backpropagation_cnn(self):

        flagStepSizeChange = 1
        self.trainingLossArray = []
        self.validationLossArray = []
        for ele1 in np.arange(self.mlffnn.epochs):
            timeEpochStart = time.time()
            if self.mlffnn.modeGradDescent == 'online':
                self.stochastic_gradient_descent_cnn()

            elif self.mlffnn.modeGradDescent == 'batch':
                self.batch_gradient_descent_cnn()

            elif self.mlffnn.modeGradDescent == 'mini_batch':
                self.mini_batch_gradient_descent_cnn()
                # print('Weights of last FC layer {0} after epoch {1}/{2}'.format(self.mlffnn.weightMatrixList[-1],ele1+1,self.mlffnn.epochs))

            if (self.validationData.shape[-1] != 0): # There is some validation data to test model
                timeStartValidation = time.time()
                self.model_validation_cnn()
                timeEndValidation = time.time()
                timeValidation = (timeEndValidation - timeStartValidation)
                # print('Validation time for epoch {0}/{1} is {2:.2f} secs'.format(ele1+1, self.mlffnn.epochs, timeValidation))
                # print('Epoch: {0}/{1}'.format(ele1+1, self.mlffnn.epochs))
                print('\ntrain_loss: {0:.1f}, val_loss: {1:.1f}, train_accuracy: {2:.1f}, val_accuracy: {3:.1f}'.format(self.trainingLoss, self.validationLoss, self.trainAccuracy, self.validationAccuracy))

                if ((self.trainAccuracy > 80) and (self.validationAccuracy > 80) and (flagStepSizeChange == 1)): # Ideally it should be ((self.trainAccuracy > 90) and (self.validationAccuracy > 90)
                    self.mlffnn.stepsize = self.mlffnn.stepsize/10 # Make step size smaller when achieving higher accuracy > 90%
                    flagStepSizeChange = 0

                if ((self.trainAccuracy > 95) and (self.validationAccuracy > 95)):
                    break
            else: # There is no validation data to test model
                print('Epoch: {0}/{1}, train_loss: {2:.1f}'.format(ele1+1, self.mlffn.epochs, self.trainingLoss))

            timeEpochEnd = time.time()
            timeEachEpoch = (timeEpochEnd - timeEpochStart)/60
            print('Time taken for epoch {0}/{1} = {2:.2f} min'.format(ele1+1, self.mlffnn.epochs, timeEachEpoch))



    def compute_train_loss_acc_cnn(self):

        """ Compute training loss and accuracy on the training data again with the weights obtained at the end of each epoch"""

        numTrainingData = self.trainData.shape[3]
        if (self.runCNNCPU == True):
            self.forwardpass_cnn(self.trainData,'test') # Compute forward pass output on the entire training data after each epoch
            self.trainingLoss = self.mlffnn.compute_loss_function(self.trainDataLabels)
            self.trainingLossArray.append(self.trainingLoss) # Keep appending the cost/loss function value for each epoch
            self.mlffnn.get_accuracy(self.trainDataLabels, self.predictedOutputcnn)
            self.trainAccuracy = self.mlffnn.accuracy
        else:
            if (self.parallelizeAcrossData == True):
                numBatches = int(np.ceil(numTrainingData/self.mlffnn.batchsize))
                batchsize = self.mlffnn.batchsize
            else:
                numBatches = numTrainingData
                batchsize = 1

            startIndex = 0
            predictedOutputAllTrainData = np.zeros(self.trainDataLabels.shape,dtype=np.float32)
            for ele in range(numBatches):
                if (startIndex+batchsize <= numTrainingData):
                    trainDataSample = self.trainData[:,:,:,startIndex:startIndex+batchsize]
                else:
                    trainDataSample = self.trainData[:,:,:,startIndex::]

                self.forwardpass_cnn(trainDataSample,'test')

                if (startIndex+batchsize <= numTrainingData):
                    predictedOutputAllTrainData[:,startIndex:startIndex+batchsize] = self.predictedOutputcnn
                else:
                    predictedOutputAllTrainData[:,startIndex::] = self.predictedOutputcnn

                startIndex += batchsize

            self.mlffnn.predictedOutput = predictedOutputAllTrainData
            self.trainingLoss = self.mlffnn.compute_loss_function(self.trainDataLabels)
            self.trainingLossArray.append(self.trainingLoss) # Keep appending the cost/loss function value for each epoch
            self.mlffnn.get_accuracy(self.trainDataLabels, self.mlffnn.predictedOutput)
            self.trainAccuracy = self.mlffnn.accuracy



    def backprop_poollayer(self,errorGradients, poolInds, poolProperties, shapeLayerLPrePooling):

        # numChannelsPrePooling = shapeLayerLPrePooling[0]
        # heightPrePooling = shapeLayerLPrePooling[1]
        # widthPrePooling = shapeLayerLPrePooling[2]
        errorGradientsPrePool = np.zeros(shapeLayerLPrePooling,dtype=np.float32)

        errorGradientnumChannels = errorGradients.shape[0]
        errorGradientsHeight = errorGradients.shape[1]
        errorGradientsWidth = errorGradients.shape[2]
        numDataPoints = errorGradients.shape[3]

        poolSize, poolStride, poolType = poolProperties
        """ Right now coded for only max pool. Will extend to avg pool as well"""
        for ele1 in range(numDataPoints):
            for ele in range(errorGradientnumChannels):
                for y in range(0, errorGradientsHeight):
                    for x in range(0, errorGradientsWidth):
                        ind = poolInds[ele,y,x, ele1]
                        ind2d = np.unravel_index(ind,(poolSize,poolSize))
                        errorGradientsPrePool[ele,y*poolStride+ind2d[0],
                                              x*poolStride+ind2d[1], ele1] = errorGradients[ele,y,x, ele1]


        return errorGradientsPrePool



    def cnn_convolve2d(self,inputImage3d, kernelFunctions,convMode='valid'):
        """ Currently wirtten only for valid mode of convolution"""
        inputHeight = inputImage3d.shape[1]
        inputWidth = inputImage3d.shape[2]
        numDataPoints = inputImage3d.shape[3]
        numKernels = kernelFunctions.shape[0]
        numChannels = kernelFunctions.shape[3]
        kernelHeight = kernelFunctions.shape[1]
        kernelWidth = kernelFunctions.shape[2]
        if convMode == 'valid':
            outputHeight = inputHeight - kernelHeight + 1 # For "valid" mode of convolution
            outputWidth = inputWidth - kernelWidth + 1 # For "valid" mode of convolution
        elif convMode == 'full':
            outputHeight = inputHeight + kernelHeight - 1 # For "full" mode of convolution
            outputWidth = inputWidth + kernelWidth - 1 # For "full" mode of convolution

        convOutput = np.zeros((numKernels,outputHeight,outputWidth,numDataPoints),dtype=np.float32)
        for ele3 in range(numDataPoints):
            for ele1 in range(numKernels):
                for ele2 in range(numChannels):
                    convOutput[ele1,:,:,ele3] += convolve2d(inputImage3d[ele2,:,:,ele3], kernelFunctions[ele1,:,:,ele2], mode=convMode)

                    # As per tensorflow bit matching, peform correlation and not convolution as below
                    # convOutput[ele1,:,:,ele3] += convolve2d(inputImage3d[ele2,:,:,ele3], np.flip(kernelFunctions[ele1,:,:,ele2],axis=(0,1)), mode=convMode)


        return convOutput



    def cnn_backward_convolve2d(self,inputImage3d, kernelFunctions,convMode='valid'):
        """ Currently wirtten only for valid mode of convolution"""
        inputHeight = inputImage3d.shape[1]
        inputWidth = inputImage3d.shape[2]
        numDataPoints = inputImage3d.shape[3]
        numKernels = kernelFunctions.shape[0]
        numChannels = kernelFunctions.shape[3]
        kernelHeight = kernelFunctions.shape[1]
        kernelWidth = kernelFunctions.shape[2]
        if convMode == 'valid':
            outputHeight = inputHeight - kernelHeight + 1 # For "valid" mode of convolution
            outputWidth = inputWidth - kernelWidth + 1 # For "valid" mode of convolution
        elif convMode == 'full':
            outputHeight = inputHeight + kernelHeight - 1 # For "full" mode of convolution
            outputWidth = inputWidth + kernelWidth - 1 # For "full" mode of convolution

        convOutput = np.zeros((numChannels,outputHeight,outputWidth,numDataPoints),dtype=np.float32)
        for ele3 in range(numDataPoints):
            for ele1 in range(numChannels):
                for ele2 in range(numKernels):
                    convOutput[ele1,:,:, ele3] += convolve2d(inputImage3d[ele2,:,:,ele3], kernelFunctions[ele2,:,:,ele1], mode=convMode)


        return convOutput




    def cnn_gradient_convolve2d(self,outputLayerL, errorConvLayerLplu1,convMode='valid'):
        """ Currently written only for valid mode of convolution"""

        numChannels = outputLayerL.shape[0]
        inputHeight = outputLayerL.shape[1]
        inputWidth = outputLayerL.shape[2]
        numDataPoints = outputLayerL.shape[3] # should be same as errorConvLayerLplu1.shape[3]
        numKernels = errorConvLayerLplu1.shape[0]
        kernelHeight = errorConvLayerLplu1.shape[1]
        kernelWidth = errorConvLayerLplu1.shape[2]
        if convMode == 'valid':
            outputHeight = inputHeight - kernelHeight + 1 # For "valid" mode of convolution
            outputWidth = inputWidth - kernelWidth + 1 # For "valid" mode of convolution
        elif convMode == 'full':
            outputHeight = inputHeight + kernelHeight - 1 # For "full" mode of convolution
            outputWidth = inputWidth + kernelWidth - 1 # For "full" mode of convolution

        convOutput = np.zeros((numKernels,outputHeight,outputWidth,numChannels, numDataPoints),dtype=np.float32)
        for ele3 in range(numDataPoints):
            for ele1 in range(numKernels):
                for ele2 in range(numChannels):
                    convOutput[ele1,:,:,ele2,ele3] = convolve2d(outputLayerL[ele2,:,:,ele3], errorConvLayerLplu1[ele1,:,:,ele3], mode=convMode)

        return convOutput






    def pooling(self,image3d, poolLayer):

        numChannels, imageHeight, imageWidth, numDataPoints = image3d.shape
        poolSize, poolStride, poolType = poolLayer

        outputHeight = (imageHeight - poolSize) // poolStride + 1
        outputWidth = (imageWidth - poolSize) // poolStride + 1
        poolingOutput = np.zeros((numChannels, outputHeight, outputWidth, numDataPoints),dtype=np.float32)

        if (poolType == 'maxpool'):
            self.maxPoolingIndex = np.zeros((numChannels, outputHeight, outputWidth, numDataPoints),dtype=np.int32) # Required for backpropagating error for maxpool. Not required for avg pool
            for ele1 in range(numDataPoints):
                for ele in range(numChannels):
                        poolingOutput[ele,:,:,ele1], self.maxPoolingIndex[ele,:,:, ele1] = self.max_pooling(image3d[ele,:,:,ele1], pool_size=poolSize, stride=poolStride)

        return poolingOutput



    def max_pooling(self, image, pool_size=2, stride=2):
        """ This function borrowed from chat gpt. So verify this once"""
        """ Log the index of maxpool as well"""
        image_height, image_width = image.shape
        output_height = (image_height - pool_size) // stride + 1
        output_width = (image_width - pool_size) // stride + 1

        output = np.zeros((output_height, output_width),dtype=np.float32)
        maxInd = np.zeros((output_height, output_width),dtype=np.int32)
        for y in range(0, output_height):
            for x in range(0, output_width):
                region = image[y*stride:y*stride+pool_size, x*stride:x*stride+pool_size]
                output[y, x] = np.max(region)
                maxInd[y, x] = np.argmax(region) # Needs to be unraveled

        return output, maxInd




