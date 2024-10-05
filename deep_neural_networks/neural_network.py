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
"""

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
np.random.seed(0)
from scipy.signal import convolve2d
import time as time

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
                if ((self.trainAccuracy > 90) and (self.validationAccuracy > 90)):
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
1. Exploding number of ANN parameters/weights especially when the size of input image is very large and we have to flatten for the ANN
2. Larger the input size, more the number of weights/parameters --> more number of examples
required to train the network.
3. For an ANN, an image with cat at top left corner of image is different from an image with cat at
bottom right of the image, so it treats it as two different outputs. Whereas, a CNN does a
local weighting/activation and hence for a CNN, both the images are treated the same.

Hence we will move to CNNS for image datasets.

1. Ensure size of kernel at any stage is smaller than size of input. Handle this gracefully, else it might crash
2. Need to keep track of index during max pool operation
3. Currently, this will support online mode of Gradient descent
4. ita is not stored for 1st layer of dense layer. Check if this is required
5. Generating/Accessing itaLastConvLayerPostPool is not correct![Fixed]
6. All pool layers have to be same type. Either all have to be maxpool or all have to be avg pool(for now). I'm not handling a mix of max and avg pools in this script
7. Write a separate function for maxpool survived itaL[Not needed. ]
8. Backpropagation for bias also needs to be done[Done]
9. Rename outputEachConvlayer to outputEachConvlayerPostPool
10. Implement update for weights and bias [Done]
11. Not clear how to perform derivative of activation function (f'(ita)) for avg pooling. Especially for activation functions other than ReLU
12. ele4 index for backwardpass_cnn. Is this looping correctly over itaConv, conVLayrs, etc?
13. Check if the code runs when there is no dense layer at all. I think, currently, I have not made a provision to cater to zero/no dense layers. This may be required!
14. inputShape which is currently argument to the CNN init is not defined for multile/batch mode
15. Define what should be the shape of data i.e channels x h x w x numData? or other way round
16. Currently not enabled for multple data points at once! This needs to enabled asap to see model accuracy/valdation, etc on the trained weight parameters
17. Verify/validate batch and mini batch gradient descent
18. Multiple definitions of numTrainData in mini_batch_gradient_desc and mini_batch_gradient_desc_cnn
19. Change the trainAccuracy and ValidationAccuracy exit condition to 95%
20. Make step size smaller and smaller as the training and validation accuracy goes beyond 90% and you wish to achieve a better accuracy
21. Add the total time of CNN execution including training and testing
"""

class ConvolutionalNeuralNetwork():

    def __init__(self, inputShape, convLayer, poolLayer, denseLayer, outputLayer):

        self.inputShape = inputShape # Numchannles, l, w
        #(#filters, size of kernel(length), activation function)
        self.convLayer = convLayer # Len of this list is the number of convolutional layers
        self.poolLayer = poolLayer # size,stride, type of pool. There will be a pooing layer for every convolutional layer.
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

        for ele in range(self.numConvLayers):
            """ Conv layer"""
            numFilters = self.convLayer[ele][0]
            filterSize = self.convLayer[ele][1] # FilterSize/KernelSize
            kernelWeights = np.random.randn(numFilters,filterSize,filterSize,inputDepth)
            bias = np.random.randn(numFilters)
            self.kernelWeights.append(kernelWeights)
            self.bias.append(bias)
            inputDepth = numFilters
            inputHeight, inputWid = inputHeight-filterSize+1, inputWid-filterSize+1 # Post "valid" convolution
            """ Pooling layer"""
            poolSize = self.poolLayer[ele][0]
            poolStride = self.poolLayer[ele][1]
            """ Verify below formula once"""
            inputHeight, inputWid = (inputHeight-poolSize)//poolStride + 1, (inputWid-poolSize)//poolStride + 1

        numNodesPostFlatten = (inputHeight * inputWid * inputDepth)#.astype(np.int32)
        flattenLayer = [(numNodesPostFlatten,'Identity')]
        denseLayerArchitecture = flattenLayer + self.denseLayer + self.outputLayer
        # self.numDenseLayers = len(self.denseLayerArchitecture)
        """ Dense layer weights initialization"""
        self.mlffnn = MLFFNeuralNetwork(denseLayerArchitecture)



    def forwardpass_cnn(self, trainDataImage):
        """ Check indices of ele3 for forward pass"""
        """ Forward pass CNN"""
        layerLOutput = trainDataImage
        numTrainingSamples = trainDataImage.shape[3] # 1st dim is numChannels, 2nd and 3rd dim are height, width, 4th dim is number of such images
        self.ItaConv = []
        self.outputEachConvlayer = []
        self.maxPoolingIndexEachConvLayer = []

        """Input layer"""
        # ele3 = 0, # Input/1st layer is taken as a dummy convolution layer but with no convolution
        layerLminus1Output = layerLOutput
        self.maxPoolingIndex = np.zeros(layerLOutput.shape,dtype=np.int32)
        self.outputEachConvlayer.append(layerLminus1Output) # Output for each layer. Stored post pooling
        self.maxPoolingIndexEachConvLayer.append(self.maxPoolingIndex)
        # ele3 is looping over the convolution layers
        for ele3 in range(1,self.numConvLayers+1):
            weightMatrixLayerLminus1toL = self.kernelWeights[ele3-1]
            """ Convolution followed by pooling"""
            """ Currently written only for valid mode of convolution"""
            itaLayerL = self.cnn_convolve2d(layerLminus1Output, weightMatrixLayerLminus1toL)
            itaLayerL += self.bias[ele3-1][:,None,None,None]
            activationFn = self.convLayer[ele3-1][2] # Activation function name
            layerLOutput = self.mlffnn.activation_function(itaLayerL,activationFn) # gives output of the activation function for the ita input
            """ Pooling"""
            poolLayer = self.poolLayer[ele3-1]
            layerLminus1Output = self.pooling(layerLOutput,poolLayer)

            self.ItaConv.append(itaLayerL) # ita is not stored for input layer. It is stored for all other layers.
            self.outputEachConvlayer.append(layerLminus1Output) # Output for each layer. Stored post pooling
            self.maxPoolingIndexEachConvLayer.append(self.maxPoolingIndex)


        numChannelsLastConvLayer = layerLminus1Output.shape[0]
        heightLastConvLayer = layerLminus1Output.shape[1]
        widthLastConvLayer = layerLminus1Output.shape[2]
        layerLminus1Output2d = np.transpose(layerLminus1Output,(3,0,1,2)).reshape(numTrainingSamples,
                                                           numChannelsLastConvLayer*heightLastConvLayer*widthLastConvLayer)


        flattenOutputConvLayers = layerLminus1Output2d.T # For batch/mini batch mode, we will have to make it 2 D and not flatten as 1D
        self.mlffnn.forwardpass(flattenOutputConvLayers)
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
        errorLastConvLayerPrePool = self.backprop_poollayer(errorLastConvLayerPostPool, poolInds, poolProperties, shapeLayerLPrePooling)

        """ First backpropagate error from pooling layer and then multiply with derivative of activation function"""
        itaLayerL = self.ItaConv[-1]
        activationFn = self.convLayer[-1][2] # Activation fn of last convolutional layer
        activationFnDerivative = self.mlffnn.derivative_activation_function(itaLayerL,activationFn)
        errorLayerLplus1 = errorLastConvLayerPrePool * activationFnDerivative

        self.errorEachConvLayer = []
        # ele4 loop goes from layer L-1(output) to layer 0 input
        for ele4 in range(self.numConvLayers-1,0,-1):
            kernelWeightsLayerL = self.kernelWeights[ele4]
            """ Below convolution should be full correlation"""
            kernelWeightsLayerLFlipHeightWidth = np.flip(kernelWeightsLayerL,axis=(1,2))
            errorLayerL = self.cnn_backward_convolve2d(errorLayerLplus1, kernelWeightsLayerLFlipHeightWidth, convMode='full')

            itaLayerL = self.ItaConv[ele4-1]
            activationFn = self.convLayer[ele4-1][2]
            activationFnDerivative = self.mlffnn.derivative_activation_function(itaLayerL,activationFn)
            """ First backpropagate error from pooling layer and then multiply with derivative of activation function"""
            shapeLayerLPrePooling = itaLayerL.shape
            poolProperties = self.poolLayer[ele4-1]
            poolInds = self.maxPoolingIndexEachConvLayer[ele4]#self.maxPoolingIndexEachConvLayer[ele4-1]
            errorLayerLPrePool = self.backprop_poollayer(errorLayerL, poolInds, poolProperties, shapeLayerLPrePooling)
            errorLayerLplus1 = errorLayerLPrePool * activationFnDerivative

            self.errorEachConvLayer.append(errorLayerLplus1) # These error arrays are packed from layer L-1 down to 1 and not from 1 to L-1. They are arranged in reverse order. (layers start from 0 to L-1)
            # Errors/delta obtained for all the layers from 2 to L



    def compute_forward_backward_pass_cnn(self, trainDataSample, trainDataLabel):

        """ Forward pass"""
        self.forwardpass_cnn(trainDataSample)

        """ Cost function computation"""
        self.costFunctionValue = self.mlffnn.compute_loss_function(trainDataLabel)

        """ Backward pass"""
        self.backwardpass_cnn(trainDataLabel)

        """ Update weights"""
        self.update_weights_cnn()



    def update_weights_cnn(self):
        # Errors/delta obtained for all the layers from 2 to L
        # Compute gradient wrt Wl_ij
        # dJ/dWl_ij = deltal+1_j * flip(yi_l)
        # gradient # dJ/dwij for all i,j

        """ Currently this dJ/dWl_ij  is computed per training sample or online mode of GD. In future, I will extend to batch and mini batch mode of GD"""
        count = -1
        for ele4 in range(self.numConvLayers-1):
            gradientCostFnwrtKernelWeights = self.cnn_gradient_convolve2d(np.flip(self.outputEachConvlayer[ele4],axis=(1,2)), self.errorEachConvLayer[count], convMode='valid')
            gradientCostFnwrtKernelWeightsAllDataPoints = np.sum(gradientCostFnwrtKernelWeights,axis=-1)
            self.kernelWeights[ele4] = self.kernelWeights[ele4] - self.mlffnn.stepsize*gradientCostFnwrtKernelWeightsAllDataPoints # Gradient descent step
            gradientCostFnwrtBias = np.sum(self.errorEachConvLayer[count],axis=(1,2,3))
            self.bias[ele4] = self.bias[ele4] - self.mlffnn.stepsize*gradientCostFnwrtBias # Gradient descent step
            count -= 1

        self.mlffnn.update_weights()



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
            self.compute_forward_backward_pass_cnn(trainDataSample,trainDataLabel)
            # print('Training example {}/{}'.format(count,numTrainData))
            count += 1

        # print('Im here')
        """ Training loss and accuracy post each epoch"""
        """ This needs batch processing which I havent brought up yet"""
        self.compute_train_loss_acc_cnn()


    def batch_gradient_descent_cnn(self):

        trainDataSample = self.trainData
        trainDataLabel = self.trainDataLabels

        self.compute_forward_backward_pass_cnn(trainDataSample,trainDataLabel)

        """ Training loss and accuracy post each epoch"""
        self.compute_train_loss_acc_cnn()


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

            self.compute_forward_backward_pass_cnn(trainDataSample,trainDataLabel)

            startIndex += self.mlffnn.batchsize

        """ Training loss and accuracy post each epoch"""
        self.compute_train_loss_acc_cnn()


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
        self.forwardpass_cnn(testData)
        self.testDataPredictedLabels = self.predictedOutputcnn


    def backpropagation_cnn(self):

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

            if (self.validationData.shape[-1] != 0): # There is some validation data to test model
                self.model_validation_cnn()
            timeEpochEnd = time.time()
            if (self.validationData.shape[-1] != 0): # There is some validation data to test model
                print('\nEpoch: {0}/{1}'.format(ele1+1, self.mlffnn.epochs))
                print('train_loss: {0:.1f}, val_loss: {1:.1f}, train_accuracy: {2:.1f}, val_accuracy: {3:.1f}'.format(self.trainingLoss, self.validationLoss, self.trainAccuracy, self.validationAccuracy))
                if ((self.trainAccuracy > 92) and (self.validationAccuracy > 92)):
                    break
            else: # There is no validation data to test model
                print('Epoch: {0}/{1}, train_loss: {2:.1f}'.format(ele1+1, self.epochs, self.trainingLoss))
            timeEachEpoch = (timeEpochEnd - timeEpochStart)/60
            print('Time taken for epoch {0} = {1:.2f} min'.format(ele1+1, timeEachEpoch))


    def compute_train_loss_acc_cnn(self):
        """ Compute training loss and accuracy on the training data again with the weights obtained at the end of each epoch"""
        self.forwardpass_cnn(self.trainData) # Compute forward pass output on the entire training data after each epoch
        self.trainingLoss = self.mlffnn.compute_loss_function(self.trainDataLabels)
        self.trainingLossArray.append(self.trainingLoss) # Keep appending the cost/loss function value for each epoch
        self.mlffnn.get_accuracy(self.trainDataLabels, self.predictedOutputcnn)
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
                        region = (errorGradientsPrePool[ele,y*poolStride:y*poolStride+poolSize, x*poolStride:x*poolStride+poolSize, ele1]).copy()
                        ind = poolInds[ele,y,x, ele1]
                        ind2d = np.unravel_index(ind,(poolSize,poolSize))
                        region[ind2d] = errorGradients[ele,y,x, ele1]
                        errorGradientsPrePool[ele,y*poolStride:y*poolStride+poolSize,
                                              x*poolStride:x*poolStride+poolSize, ele1] = region


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
        maxInd = np.zeros((output_height, output_width),dtype=np.float32)
        for y in range(0, output_height):
            for x in range(0, output_width):
                region = image[y*stride:y*stride+pool_size, x*stride:x*stride+pool_size]
                output[y, x] = np.max(region)
                maxInd[y, x] = np.argmax(region) # Needs to be unraveled

        return output, maxInd










