# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 19:09:26 2025

@author: Sai Gunaranjan
"""


"""
Recurrent Neural network (RNN)

Implemented a vanilla stateful RNN, where the final hidden state after N timesteps feeds in as input hidden state for the
next batch of data. This is called a stateful RNN. I have also derived the back propagation algorithm to obtain the
gradients of the cost function wrt weight matrices Whh and Wxh. The derivation is available in my onenote.
I have also derived and shown how vanishing and exploding gradients is a problem in vanilla RNNs

Trained on Harry Potter one chapter of text with:
    1. batch size of 32, was able to hit train_loss: 2.2, val_loss: 3.2, train_accuracy: 56.1, val_accuracy: 42.5 after 1000 epochs
    2. batch size of 1, was able to hit train_loss: 3.0, val_loss: 3.2, train_accuracy: 38.2, val_accuracy: 36.8



25/07/2025
Run RNN successfully generating character level text

In this commit, I have fixed the error/bug that was generating random meaningless text during the prediction part of the RNN! The bug was that, during the prediction part, the input 1-hot vector was not being generated correctly. I was placing a 1 at the index of the chr2idx, however, I should place it at index chr2idx+1. This is because I'm additionally  prepending a 1 at the beginning of the input 1 hot vector (hence increasing its dimension by 1) to cater to the bias in the Wxh matrix.
I found this bug while bit matching my prediction logic with Andrej's vanilla RNN code. It was actually very useful to have a baseline script to check and verify my implementation.
Post fixing this bug, I'm now able to generate meaningful text and words similar to the prediction from Andrej's script. However, with larger learning rates like 1e-2, the code is  crashing (possibly due to exploding gradients) while evaluating the exp(x) part of the tanh function. I will examine this more closely. Following are the changes which have been made in this commit.
1. Replaced the Xavier initialization of the weights for tanh activation with weights drawn from gaussian distribution of 0 mean and sigma = 0.01. I did this as Andrej was also doing the same weight initialization. But I need to check if Xavier initialization also works and there are no exploding gradients.

2. Added debug prints (commented out) in the backward pass to check for distribution of gradients.

3. Previously, I was computing the gradients of the loss wrt Wxh and Whh as a mean across time axis which is ideally not correct. Of course it is just a scaling that can be accounted back into the learning rate. I am now computing the gradients of the loss wrt Wxh and Whh as a sum across time which means each time step of the seq_len time steps contributes to the overall loss and hence we take the gradients as a sum across all the examples in a time step.

4. Added a comment to clip the gradients to force them to not exceed +/-5. If the gradients are large, then the updated weights will aslo get large and this will result in the output ita to be large and when evaulating the exp(ita) part of the tanh, it will crash for large numbers! I need to study this more closely. I have adopted this line form Andrej's script where in he is also limiting the gradients.

5. Changed the way in which I predict a text after each epoch. Previously I was always force feeding hidden satte as 0 and input vector corrsponding to H (which is the first character of the training text). Now, I randomly select the starting character of each batch of examples for each epoch as the input vector. Hidden state is anyways available from previous batch in the same epoch.

6. Also, made the batch size to 1 while training. This is how Andrejs does in his script. I need to see how to maintain continuity in input and hidden sate when dealing with batches of data. I will do this in the subsequent commits.



28/07/2025
Generalized RNN to different length hidden state vector for each layer


In this commit, I have generalized the RNN architecture to cater to different sizes of the hidden state vector
for each RNN layer. Previously, due to matrix only operations, the size of the hidden state vector for each RNN layer
was forced to be same as the input size(also vocab size). But this reduces the flexibility on the hidden states.
By allowing for the hidden state vector to take on any size for each RNN layer, irrespective of the vocab size,
the RNN network can learn different kinds of long term dependencies like " " (quote unquote), etc. Since the hidden state
vector for each layer is different size, we can no longer work with matrices across layers and hence I have
introduced lists across layers. But within a layer, across time steps, we still deal with matrices. However,
the execution time now increases from 1.2 seconds per epoch to 12 seconds per epoch since there are list operations.
The code is running fine. I still need to incorporate the checks for gradient exploding. Moreover, I want to spend some time
understanding and appreciating the problem of exploding gradients in RNNs and not necessarily only from a code crashing stand point.
There's another issue. When I increase the batch size from 1 to say 32, the code still runs fine, but the RNN
doesnt seem to generate meaningful words. This could possibly be because of the following reason.
Since the training data is divided into 32 batches (if batch size is selected to 32), each batch might be starting randomly
and not necessarily from the beginning of a sentence. Hence, the network might not be learning meaningful
long term dependencies. This is my hunch. I need to verify this though! I also need to understand the validation loss and accuracy
for text data.


Tasks:
    1. Understand impact of exploding gradients.
    2. Understand how to define validation loss and accuracy in the contect of text generation
    3. Understand RNN behaviour when dealing with batches of training text.
    4. Implement gradient clipping
    5. Check if we are witnessing vanishing and exploding gradients. In RNNs, exploding gradients problem can be
    handled by clipping but what do we do about vanishing gradients? LSTMs/GRUs are the solutions. These variants
    of RNNs avoid gradient vanishing by providing gradient flow paths!
    6. Understand the importance of sequence length. How to choose sequence length?
    7. Visualize the vanishing gradients across time steps.


Interesting blog on RNNs by Andrej Karpathy:
    http://karpathy.github.io/2015/05/21/rnn-effectiveness/
and aslo a video on RNNs, LSTMs by ANdrej:
    https://www.youtube.com/watch?v=yCC09vCHzF8

Good stanford lecture on RNNs, LSTM, GRU:
    https://www.youtube.com/watch?v=6niqTuYFZLQ&t=8s

and also lectures by Mitesh Khapra on Deep learning

I have used the above resourses for understanding RNNs.

Regularization is not used often in RNNs/LSTMs, etc.

"""

import numpy as np
import time as time
np.random.seed(1)
from textfile_preprocessing import prepare_data, get_batch
# np.seterr(over='raise')



class RecurrentNeuralNetwork():

    def __init__(self, inputShape, numRNNLayers, outputShape, numTimeSteps):

        # We will assume output shape to be same as input shape, since both input and output are characters or words
        self.inputShape = inputShape
        self.numRNNLayers = numRNNLayers
        self.outputShape = outputShape
        self.numTimeSteps = numTimeSteps
        self.numTotalLayers = 1 + self.numRNNLayers + 1 # (1 for input, 1 for output, numRNNLayers)

        """ Weight initialization"""
        """ Xavier method of weight initialization for tanh activation for vanilla RNN"""
        self.hiddenShape = np.random.randint(self.inputShape, self.inputShape+50, size=self.numRNNLayers+1) # Length of hidden state vector can be different for each layer
        self.hiddenShape[-1] = self.outputShape # Final output layer hidden state vector is a 0 vector of shape vocab size
        self.inputShapeEachLayer = np.zeros((self.numRNNLayers+2,),dtype=np.int32) # Size of input to each layer (rnn layer and final output layer)
        self.inputShapeEachLayer[0] = self.inputShape

        fanInWxh = self.inputShape+1 # inputShape+1 absorbs the bias term to the Wxh matrix
        self.Wxh = []
        self.Whh = []
        for ele in range(self.numRNNLayers+1):
            fanInWhh = fanOutWhh = self.hiddenShape[ele]
            fanOutWxh = fanOutWhh
            scalingFactorXavierInitWhh = np.sqrt(2/(fanInWhh+fanOutWhh)) * 5/3
            scalingFactorXavierInitWxh = np.sqrt(2/(fanInWxh+fanOutWxh)) * 5/3

            if (ele == self.numRNNLayers):
                Whh = np.zeros((fanOutWhh, fanInWhh),dtype=np.float32) # Final output layer doesnt have a hidden state input and hence Whh for final layer is 0
            else:
                Whh = np.random.randn(fanOutWhh, fanInWhh) * scalingFactorXavierInitWhh
            Wxh = np.random.randn(fanOutWxh, fanInWxh) * scalingFactorXavierInitWxh # self.inputShape+1 absorbs the bias term to the Wxh matrix, self.numRNNLayers+1 is for the final output layer
            fanInWxh = fanOutWhh + 1 # fanOutWhh+1 absorbs the bias term to the Wxh matrix
            self.inputShapeEachLayer[ele+1] = fanOutWhh

            self.Wxh.append(Wxh)
            self.Whh.append(Whh)

        # Wxh and Whh matrices size vary with each layer



    def set_model_params(self, batchsize = 32, epochs = 100000, stepsize = 0.1):
        """ For RNN, we default to batch mode of gradient descent and categorical cross entropy cost function"""

        self.epochs = epochs
        self.stepsize = stepsize
        self.batchsize = batchsize # Batch size is typically a power of 2


    def preprocess_textfile(self,textfile):

        self.params = prepare_data(textfile, n_segments = self.batchsize, seq_len = self.numTimeSteps)



    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def tanh(self,z):
        # tanh = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

        # try:
        #     ePowpz = np.exp(z)
        # except FloatingPointError:
        #     print("Overflow detected!")

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



    def define_hiddenstatematrix(self,hiddenState):
        self.hiddenStateMatrix = []
        for ele1 in range(self.numRNNLayers+1):
        	temp = np.zeros((self.numTimeSteps+1, self.hiddenShape[ele1], self.batchsize),dtype=np.float32)
        	temp[0,:,:] = hiddenState[ele1] # Write the hidden state at the final time step back to the inital time step for next batch
        	self.hiddenStateMatrix.append(temp)


    def define_inputmatrix(self,trainDataSample):
        self.inputMatrixX = []
        for layer in range(self.numRNNLayers + 2):
        	cell = np.ones((self.numTimeSteps, self.inputShapeEachLayer[layer]+1, self.batchsize), dtype=np.float32) ## self.inputShape+1 is for the bias
        	if layer == 0:
        		cell[:,1::, :] = trainDataSample # 1st term is for the bias term
        	self.inputMatrixX.append(cell)


    def define_outputmatrix(self):
        self.outputMatrix = []
        for layer in range(self.numRNNLayers+1):
        	cell = np.zeros((self.numTimeSteps, self.inputShapeEachLayer[layer+1], self.batchsize), dtype=np.float32)
        	self.outputMatrix.append(cell)



    def forwardpass_rnn(self, trainDataSample, hiddenState):

        # numTrainingSamples = trainDataSample.shape[2] # batch size, 1st dimension is sequence length, 2nd dimension is length of vector, 3rd dimension is batch size

        self.define_hiddenstatematrix(hiddenState)

        self.define_inputmatrix(trainDataSample)

        self.define_outputmatrix()

        self.ita = [row.copy() for row in self.outputMatrix] # ita is same size as output matrix

        for ele2 in range(self.numTimeSteps):
            for ele1 in range(self.numRNNLayers+1): # +1 is for the final output hidden layer with softmax
                hiddenStateTminus1LayerLplus1 = self.hiddenStateMatrix[ele1][ele2,:,:]
                hiddenStateTLayerL = self.inputMatrixX[ele1][ele2,:,:]
                itaLayerLPlus1 = (self.Whh[ele1] @ hiddenStateTminus1LayerLplus1) + (self.Wxh[ele1] @ hiddenStateTLayerL)
                if (ele1 != self.numRNNLayers): # All RNN layers have tanh/sigmoid activatio fn. Last layer has softmax activation fn
                    hiddenStateTLayerLplus1 = self.activation_function(itaLayerLPlus1, 'tanh')
                else:
                    hiddenStateTLayerLplus1 = self.activation_function(itaLayerLPlus1, 'softmax')

                self.ita[ele1][ele2,:,:] = itaLayerLPlus1
                self.inputMatrixX[ele1+1][ele2,1::,:] = hiddenStateTLayerLplus1

                if (ele1 != self.numRNNLayers):
                    self.hiddenStateMatrix[ele1][ele2+1,:,:] = hiddenStateTLayerLplus1
                else:
                    self.hiddenStateMatrix[ele1][ele2+1,:,:] = self.hiddenStateMatrix[ele1][ele2,:,:] # carry forward the zeros from last output layer of previous time step

                self.outputMatrix[ele1][ele2,:,:] = hiddenStateTLayerLplus1


        hiddenState = [row[-1,:,:] for row in self.hiddenStateMatrix] # hidden state of each layer and last time step is stored and used as input hidden state of next charcter

        return hiddenState


    def backwardpass_rnn(self,trainDataLabel):
        # trainDataLabel should be of shape numTimeSteps, outputShape, batch size

        #Errors/delta = dL/d ita
        self.errorMatrix = [row.copy() for row in self.outputMatrix]
        for ele2 in range(self.numTimeSteps-1,-1,-1):
            for ele1 in range(self.numRNNLayers,-1,-1):
                if (ele1 == self.numRNNLayers):
                    """ Final output layer for each time step"""
                    self.errorMatrix[ele1][ele2,:,:] = self.outputMatrix[ele1][ele2,:,:] - trainDataLabel[ele2,:,:] # (y - d) For softmax activation function with categorical cross entropy cost function. Used for classification tasks.
                else:
                    itaLayerL = self.ita[ele1][ele2,:,:]
                    activationFn = 'tanh'
                    activationFnDerivative = self.derivative_activation_function(itaLayerL,activationFn)
                    if (ele2 == self.numTimeSteps-1):
                        self.errorMatrix[ele1][ele2,:,:] = (self.Wxh[ele1+1][:,1::].T @ self.errorMatrix[ele1+1][ele2,:,:]) * activationFnDerivative
                    else:
                        self.errorMatrix[ele1][ele2,:,:] = ((self.Wxh[ele1+1][:,1::].T @ self.errorMatrix[ele1+1][ele2,:,:]) +
                                                      (self.Whh[ele1].T @ self.errorMatrix[ele1][ele2+1,:,:])) * activationFnDerivative

        # Need to change below lines for 2d lists of 2d arrays
        # np.clip(self.errorMatrix, -5, 5, out=self.errorMatrix) # Clip to prevent exploding gradients

        # plt.hist(self.errorMatrix[0,:,:,0].flatten(),bins=50)
        # print('\n\n')
        # print('Min value of gradient: {0:.1f}'.format(np.amin(self.errorMatrix[0,:,:,0].flatten())))
        # print('Max value of gradient: {0:.1f}'.format(np.amax(self.errorMatrix[0,:,:,0].flatten())))
        # percentile = 95
        # print('{0} percentile value of gradient: {1:.1f}'.format(percentile, np.percentile(np.abs(self.errorMatrix[0,:,:,0].flatten()),percentile)))
        # print('--')

    def update_weights_rnn(self):


        batchSize = self.errorMatrix[0].shape[-1]

        # 1. Slice layers from inputMatrixX (exclude last layer)
        outputEachLayer = [row.copy() for row in self.inputMatrixX[0:self.numRNNLayers+1]] # Last layer is the final output which we are not interested in!


        # 3. Batch-wise matrix multiply errorMatrix and outputEachLayer, divide by batchSize. # Division is to take mean across gradients of a batch
        tempGradientsWxh = [
        		(self.errorMatrix[i] @ np.transpose(outputEachLayer[i],(0,2,1))) / batchSize
        	for i in range(len(self.errorMatrix))
        ]

        # 4. Sum tempGradientsWxh across time steps (axis=1)
        gradientCostFnwrtWxh = [np.sum(tempGradientsWxh[i],axis=0) for i in range(len(tempGradientsWxh))]
        # gradientCostFnwrtWxh = [np.mean(tempGradientsWxh[i],axis=0) for i in range(len(tempGradientsWxh))]


        # Below line needs to be modified for lists
        # np.clip(gradientCostFnwrtWxh, -5, 5, out=gradientCostFnwrtWxh)

        # 6. Gradient descent update
        self.Wxh = [W - self.stepsize * grad for W, grad in zip(self.Wxh, gradientCostFnwrtWxh)]




        # 7. Slice hiddenStateMatrix for all layers and only first numTimeSteps
        hiddenStateEachLayer = [row[0:self.numTimeSteps,:,:].copy() for row in self.hiddenStateMatrix]


        # 9. Batch-wise matrix multiply errorMatrix and hiddenStateEachLayer, divide by batchSize. # Division is to take mean across gradients of a batch
        tempGradientsWhh = [
                (self.errorMatrix[i] @ np.transpose(hiddenStateEachLayer[i],(0,2,1))) / batchSize
            for i in range(len(self.errorMatrix))
        ]

        # 10. Sum tempGradientsWhh across time steps
        gradientCostFnwrtWhh = [np.sum(tempGradientsWhh[i],axis=0) for i in range(len(tempGradientsWhh))]
        # gradientCostFnwrtWhh = [np.mean(tempGradientsWhh[i],axis=0) for i in range(len(tempGradientsWhh))]


        # Below line needs to be modified for lists
        # np.clip(gradientCostFnwrtWhh, -5, 5, out=gradientCostFnwrtWhh)

        # 12. Gradient descent update
        self.Whh = [W - self.stepsize * grad for W, grad in zip(self.Whh, gradientCostFnwrtWhh)]





    def compute_forward_backward_pass_rnn(self, trainDataSample, trainDataLabel, hiddenState):

        """ Forward pass"""
        t1 = time.time()
        hiddenState = self.forwardpass_rnn(trainDataSample, hiddenState)
        # predictedOutput = self.outputMatrix[-1,:,:,:]
        t2 = time.time()
        # print('Time taken for forward pass = {0:.2f} s'.format(t2-t1))

        # """ Cost function computation. May not be required! Required only when computing loss functon for each mini batch"""
        # t3 = time.time()
        # self.costFunctionValue = self.compute_loss_function(trainDataLabel, predictedOutput)
        # t4 = time.time()
        # print('Time taken for cost fn eval = {0:.2f} s'.format(t4-t3))

        """ Backward pass"""
        t5 = time.time()
        self.backwardpass_rnn(trainDataLabel)
        t6 = time.time()
        # print('Time taken for backward pass = {0:.2f} s'.format(t6-t5))

        """ Update weights"""
        t7 = time.time()
        self.update_weights_rnn()
        t8 = time.time()
        # print('Time taken for Weight update = {0:.2f} s'.format(t8-t7))

        return hiddenState


    def backpropagation_rnn(self):

        flagStepSizeChange = 1
        self.trainingLossArray = []
        self.validationLossArray = []
        for ele1 in np.arange(self.epochs):
            timeEpochStart = time.time()

            self.mini_batch_gradient_descent_rnn()

            """ Training loss and accuracy post each epoch"""
            t3 = time.time()
            self.compute_train_loss_acc_rnn()
            t4 = time.time()
            # print('Time taken for computing training loss and accuracy after epoch = {0:.2f} min'.format(t4-t3)/60)

            """ There is always validation data to test model"""
            timeStartValidation = time.time()
            self.compute_validation_loss_acc_rnn()
            timeEndValidation = time.time()
            timeValidation = (timeEndValidation - timeStartValidation)

            print('\ntrain_loss: {0:.1f}, val_loss: {1:.1f}, train_accuracy: {2:.1f}, val_accuracy: {3:.1f}'.format(self.trainingLoss, self.validationLoss, self.trainAccuracy, self.validationAccuracy))
            # Add a prediction after each epoch just to check the performance
            if ((self.trainAccuracy > 80) and (self.validationAccuracy > 80) and (flagStepSizeChange == 1)): # Ideally it should be ((self.trainAccuracy > 90) and (self.validationAccuracy > 90)
                self.stepsize = self.stepsize/10 # Make step size smaller when achieving higher accuracy > 90%
                flagStepSizeChange = 0

            if ((self.trainAccuracy > 95) and (self.validationAccuracy > 95)):
                break


            timeEpochEnd = time.time()
            timeEachEpoch = (timeEpochEnd - timeEpochStart)/60
            print('Time taken for epoch {0}/{1} = {2:.2f} min'.format(ele1+1, self.epochs, timeEachEpoch))

            predSeqLen = 200
            self.predict(predSeqLen) # Generate a character sequence of length = predSeqLen, at the end of each epoch



    def compute_loss_function(self,trainDataLabel, predictedOutput):

        # Cost function = 'categorical_cross_entropy'
        mask = predictedOutput !=0 # Avoid 0 values in log2 evaluation. But this is not correct. It can mask wrong classifications.
        N = predictedOutput.shape[0] * predictedOutput.shape[2] # numTimeSteps * numExamples
        # cost fn = -Sum(di*log(yi))/N, where di is the actual output and yi is the predicted output, N is the batch size.
        costFunction = (-np.sum((trainDataLabel[mask]*np.log2(predictedOutput[mask]))))/N # Mean loss across data points
        """ Need to divide by N (batch size) to get the mean loss across data points"""

        return costFunction



    def train(self):

        self.backpropagation_rnn()



    def mini_batch_gradient_descent_rnn(self):

        randBatchInd = np.random.randint(0,self.params["n_train_batches"])
        """ For stateful RNN, we may not need to shuffle the data while training, I think. Will verify this!"""
        hiddenState = [np.zeros((self.hiddenShape[ele], self.batchsize), dtype=np.float32) for ele in range(self.numRNNLayers + 1)]
        for batch_step in range(self.params["n_train_batches"]):
            trainDataSample, trainDataLabel = get_batch(
                self.params["train_data_segments"],
                self.params["train_label_segments"],
                batch_step,
                self.params["seq_len"],
                self.params["vocab_size"],
            )
            if (batch_step == randBatchInd):
                self.hiddenStateForPredict = [h[:, 0] for h in hiddenState] # Sample the previous hidden state for 1 example since prediction works with 1 sample at a time
                self.startIdx = np.argmax(trainDataSample[0,0,:]) # Store the starting character idx for the next sequence starting
            t1 = time.time()
            trainDataSample = np.transpose(trainDataSample,(1,2,0))
            trainDataLabel = np.transpose(trainDataLabel,(1,2,0))
            hiddenState = self.compute_forward_backward_pass_rnn(trainDataSample,trainDataLabel, hiddenState)
            t2 = time.time()





    def compute_train_loss_acc_rnn(self):

        """ Compute training loss and accuracy on the training data again with the weights obtained at the end of each epoch
        Hidden state is set back to 0 when evaluating the training loss/accuracy after training for each epoch.
        But I could as well use the hidden state from the last example of the last time step of previous epoch!

        But within an epoch, the hidden state is carried forward across the batches and examples"""

        actualOutputAllTrainData = np.zeros((self.numTimeSteps,self.outputShape,self.batchsize, self.params["n_train_batches"]))
        predictedOutputAllTrainData = np.zeros((self.numTimeSteps,self.outputShape,self.batchsize, self.params["n_train_batches"]))
        hiddenState = [np.zeros((self.hiddenShape[ele], self.batchsize), dtype=np.float32) for ele in range(self.numRNNLayers + 1)] # Currently hidden state being rolled back to 0!
        for batch_step in range(self.params["n_train_batches"]):
            trainDataSample, trainDataLabel = get_batch(
                self.params["train_data_segments"],
                self.params["train_label_segments"],
                batch_step,
                self.params["seq_len"],
                self.params["vocab_size"],
            )

            trainDataSample = np.transpose(trainDataSample,(1,2,0))
            trainDataLabel = np.transpose(trainDataLabel,(1,2,0))
            hiddenState = self.forwardpass_rnn(trainDataSample,hiddenState)
            predictedOutputAllTrainData[:,:,:,batch_step] = self.outputMatrix[-1]
            actualOutputAllTrainData[:,:,:,batch_step] = trainDataLabel


        predictedOutputAllTrainData = predictedOutputAllTrainData.reshape(self.numTimeSteps,self.outputShape,self.batchsize*self.params["n_train_batches"])
        actualOutputAllTrainData = actualOutputAllTrainData.reshape(self.numTimeSteps,self.outputShape,self.batchsize*self.params["n_train_batches"])
        self.trainingLoss = self.compute_loss_function(actualOutputAllTrainData, predictedOutputAllTrainData)
        self.trainingLossArray.append(self.trainingLoss) # Keep appending the cost/loss function value for each epoch
        self.get_accuracy(actualOutputAllTrainData, predictedOutputAllTrainData)
        self.trainAccuracy = self.accuracy



    def compute_validation_loss_acc_rnn(self):

        """ Compute validation loss and accuracy on the validation data with the weights obtained at the end of each epoch
        Here also, hidden state is set back to 0 when evaluating the validation loss/accuracy after training for each epoch.
        But I could as well use the hidden state from the last example of the last time step of previous epoch!

        But within an epoch, the hidden state is carried forward across the batches and examples
        """


        actualOutputAllValidationData = np.zeros((self.numTimeSteps,self.outputShape,self.batchsize, self.params["n_val_batches"]))
        predictedOutputAllValidationData = np.zeros((self.numTimeSteps,self.outputShape,self.batchsize, self.params["n_val_batches"]))
        hiddenState = [np.zeros((self.hiddenShape[ele], self.batchsize), dtype=np.float32) for ele in range(self.numRNNLayers + 1)] # Currently hidden state being rolled back to 0!
        for batch_step in range(self.params["n_val_batches"]):
            validationDataSample, validationDataLabel = get_batch(
                self.params["val_data_segments"],
                self.params["val_label_segments"],
                batch_step,
                self.params["seq_len"],
                self.params["vocab_size"],
            )

            validationDataSample = np.transpose(validationDataSample,(1,2,0))
            validationDataLabel = np.transpose(validationDataLabel,(1,2,0))
            hiddenState = self.forwardpass_rnn(validationDataSample,hiddenState)
            predictedOutputAllValidationData[:,:,:,batch_step] = self.outputMatrix[-1]
            actualOutputAllValidationData[:,:,:,batch_step] = validationDataLabel


        predictedOutputAllValidationData = predictedOutputAllValidationData.reshape(self.numTimeSteps,self.outputShape,self.batchsize*self.params["n_val_batches"])
        actualOutputAllValidationData = actualOutputAllValidationData.reshape(self.numTimeSteps,self.outputShape,self.batchsize*self.params["n_val_batches"])
        self.validationLoss = self.compute_loss_function(actualOutputAllValidationData, predictedOutputAllValidationData)
        self.validationLossArray.append(self.validationLoss) # Keep appending the cost/loss function value for each epoch
        self.get_accuracy(actualOutputAllValidationData, predictedOutputAllValidationData)
        self.validationAccuracy = self.accuracy



    def get_accuracy(self, trueLabels, predLabels, printAcc=False):
        predClasses = np.argmax(predLabels,axis=1)
        actualClasses = np.argmax(trueLabels,axis=1)
        self.accuracy = np.mean(predClasses == actualClasses) * 100
        if printAcc:
            print('\nAccuracy of NN = {0:.2f} % \n'.format(self.accuracy))


    def predict(self, predSeqLen):


        textString = ''
        hiddenStateTminus1LayerLplus1 = [arr.copy() for arr in self.hiddenStateForPredict]
        idx = self.startIdx
        startingchar = self.params['idx2char'][idx]
        textString += startingchar
        inputVector = np.zeros((self.inputShape+1,)) # +1 for the bias
        inputVector[0] = 1 # 1st element is for the bias term
        inputVector[idx+1] = 1 # To account for the element 1 added at the beginning



        for ele2 in range(predSeqLen):

            for ele1 in range(self.numRNNLayers+1):

                ita = (self.Whh[ele1] @ hiddenStateTminus1LayerLplus1[ele1]) + (self.Wxh[ele1] @ inputVector)

                if (ele1 != self.numRNNLayers): # All RNN layers have tanh/sigmoid activatio fn. Last layer has softmax activation fn
                    hiddenStateTLayerLplus1 = self.activation_function(ita, 'tanh')
                    inputVector = np.concatenate(([1],hiddenStateTLayerLplus1)) # Prepended with 1 to take care of bias
                    hiddenStateTminus1LayerLplus1[ele1] = hiddenStateTLayerLplus1 # Overwrite/Update the hidden state for next time step
                else:
                    # Need input for softmax to be a vector and not ndarray
                    hiddenStateTLayerLplus1 = (self.activation_function(ita[:,None], 'softmax')).squeeze()
                    inputVector = hiddenStateTLayerLplus1 # This is the final output and hence no bias term required. Also, its a pmf
                    hiddenStateTminus1LayerLplus1[ele1] = np.zeros((self.outputShape,),dtype=np.float32) # For last layer with no hidden state, set to 0

            """Input vector/output after looping through all the layers is a probability distribution over
            the vocabulary
            """
            outputPMF = inputVector.flatten()

            # Sample from this distribution
            values = np.arange(self.outputShape)
            chrIndex = np.random.choice(values, p=outputPMF)
            char = self.params['idx2char'][chrIndex]
            textString += char
            inputVector = np.zeros((self.inputShape+1,))
            inputVector[0] = 1 # 1st element is for the bias term
            inputVector[chrIndex+1] = 1 # To account for the element 1 added at the beginning

        print('Predicted text:\n',textString)





if 0:

    def train_rnn(self,trainData,trainDataLabels,split = 1):
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

        self.backpropagation_rnn()

    def mini_batch_gradient_descent_rnn(self):
        """ For stateful RNN, we may not need to shuffle the data while training, I think Will verify this!"""
        numTrainData = self.trainData.shape[3]
        arr = np.arange(numTrainData)
        # """Randomly shuffle the order of feeding the training data for each epoch"""
        # np.random.shuffle(arr)
        # """ arr is the randomly shuffled order of sampling the training data"""
        trainDataShuffle = self.trainData[:,:,:,arr]
        trainDataLabelsShuffle = self.trainDataLabels[:,arr]
        numTrainingData = self.trainData.shape[3]
        numBatches = int(np.ceil(numTrainingData/self.batchsize))
        startIndex = 0
        hiddenState = np.zeros((self.numRNNLayers+1, self.inputShape, self.batchsize),dtype=np.float32)
        for ele in range(numBatches):
            if (startIndex+self.batchsize <= numTrainingData):
                trainDataSample = trainDataShuffle[:,:,:,startIndex:startIndex+self.batchsize]
                trainDataLabel = trainDataLabelsShuffle[:,startIndex:startIndex+self.batchsize]
            else:
                trainDataSample = trainDataShuffle[:,:,:,startIndex::]
                trainDataLabel = trainDataLabelsShuffle[:,startIndex::]
            t1 = time.time()
            hiddenState = self.compute_forward_backward_pass_rnn(trainDataSample,trainDataLabel, hiddenState)
            t2 = time.time()
            # print('Time taken for batch {0}/{1} is {2:.2f} s'.format(ele+1,numBatches,(t2-t1)))

            startIndex += self.batchsize



    def compute_train_loss_acc_rnn(self):

        """ Compute training loss and accuracy on the training data again with the weights obtained at the end of each epoch"""

        numTrainingData = self.trainData.shape[3]
        numBatches = int(np.ceil(numTrainingData/self.batchsize))
        batchsize = self.batchsize
        startIndex = 0
        predictedOutputAllTrainData = np.zeros(self.trainDataLabels.shape,dtype=np.float32)
        hiddenState = np.zeros((self.numRNNLayers+1, self.inputShape, self.batchsize),dtype=np.float32)
        for ele in range(numBatches):
            if (startIndex+batchsize <= numTrainingData):
                trainDataSample = self.trainData[:,:,:,startIndex:startIndex+batchsize]
            else:
                trainDataSample = self.trainData[:,:,:,startIndex::]

            hiddenState = self.forwardpass_rnn(trainDataSample,hiddenState)

            if (startIndex+batchsize <= numTrainingData):
                predictedOutputAllTrainData[:,:,startIndex:startIndex+batchsize] = self.outputMatrix[-1,:,:,:]
            else:
                predictedOutputAllTrainData[:,:,startIndex::] = self.outputMatrix[-1,:,:,:]

            startIndex += batchsize


        self.trainingLoss = self.compute_loss_function(self.trainDataLabels, predictedOutputAllTrainData)
        self.trainingLossArray.append(self.trainingLoss) # Keep appending the cost/loss function value for each epoch
        self.get_accuracy(self.trainDataLabels, predictedOutputAllTrainData)
        self.trainAccuracy = self.accuracy



    def compute_validation_loss_acc_rnn(self):

        """ Compute training loss and accuracy on the training data again with the weights obtained at the end of each epoch"""

        numvalidationData = self.validationData.shape[3]
        numBatches = int(np.ceil(numvalidationData/self.batchsize))
        batchsize = self.batchsize
        startIndex = 0
        predictedOutputAllValidationData = np.zeros(self.validationDataLabels.shape,dtype=np.float32)
        hiddenState = np.zeros((self.numRNNLayers+1, self.inputShape, self.batchsize),dtype=np.float32)
        for ele in range(numBatches):
            if (startIndex+batchsize <= numvalidationData):
                validationDataSample = self.validationData[:,:,:,startIndex:startIndex+batchsize]
            else:
                validationDataSample = self.validationData[:,:,:,startIndex::]

            hiddenState = self.forwardpass_rnn(validationDataSample,hiddenState)

            if (startIndex+batchsize <= numvalidationData):
                predictedOutputAllValidationData[:,:,startIndex:startIndex+batchsize] = self.outputMatrix[-1,:,:,:]
            else:
                predictedOutputAllValidationData[:,:,startIndex::] = self.outputMatrix[-1,:,:,:]

            startIndex += batchsize


        self.validationLoss = self.compute_loss_function(self.validationDataLabels, predictedOutputAllValidationData)
        self.validationLossArray.append(self.validationLoss) # Keep appending the cost/loss function value for each epoch
        self.get_accuracy(self.validationDataLabels, predictedOutputAllValidationData)
        self.validationAccuracy = self.accuracy