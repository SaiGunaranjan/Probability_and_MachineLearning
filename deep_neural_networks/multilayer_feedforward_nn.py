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



Reference:
https://medium.com/@ja_adimi/neural-networks-101-hands-on-practice-25df515e13b0

"""

"""
NN for classification problem

1. Inititalize all the weights. How to initialize W matrices of each layer[Done. Random sampling from a uniform distribution]
2. Normalize input features to zero mean and unit variance, so that no one features totally dominates the output.
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
15. Compute confusion matrix
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

    def set_model_params(self,modeGradDescent = 'online',batchsize = 1, costfn = 'categorical_cross_entropy',epochs = 100000, stepsize = 0.1):
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
            mask = self.predictedOutput !=0 # Avoid 0 values in log2 evaluation
            costFunction = -np.sum((trainDataLabel[mask]*np.log2(self.predictedOutput[mask]))) # -Sum(di*log(yi)), where di is the actual output and yi is the predicted output
        elif (self.costfn == 'squared_error'):
            costFunction = 0.5*np.sum((self.predictedOutput - trainDataLabel)**2) # 1/2 Sum((yi - di)**2), where di is the actual output and yi is the predicted output

        return costFunction


    def forwardpass(self, trainDataSample):

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
        # This backward pass computation is for 'categorical_cross_entropy', 'squared error' loss functions

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



    def compute_forward_backward_pass(self, trainDataSample, trainDataLabel):

        """ Forward pass"""
        self.forwardpass(trainDataSample)

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
            gradientCostFnwrtWeights = self.errorEachLayer[count] @ self.outputEachlayer[ele4].T
            self.weightMatrixList[ele4] = self.weightMatrixList[ele4] - self.stepsize*gradientCostFnwrtWeights # Gradient descent step
            count -= 1


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
        self.forwardpass(testData)
        self.testDataPredictedLabels = self.predictedOutput



    def get_accuracy(self, trueLabels, predLabels, printAcc=False):
        predClasses = np.argmax(predLabels,axis=0)
        actualClasses = np.argmax(trueLabels,axis=0)
        self.accuracy = np.mean(predClasses == actualClasses) * 100
        if printAcc:
            print('\nAccuracy of NN = {0:.2f} % \n'.format(self.accuracy))


    def compute_train_loss_acc(self):
        """ Compute training loss and accuracy on the training data again with the weights obtained at the end of each epoch"""
        self.forwardpass(self.trainData) # Compute forward pass output on the entire training data after each epoch
        self.trainingLoss = self.compute_loss_function(self.trainDataLabels)
        self.trainingLossArray.append(self.trainingLoss) # Keep appending the cost/loss function value for each epoch
        self.get_accuracy(self.trainDataLabels, self.predictedOutput)
        self.trainAccuracy = self.accuracy















