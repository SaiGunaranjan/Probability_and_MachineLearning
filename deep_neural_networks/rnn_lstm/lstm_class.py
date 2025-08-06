# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 19:09:26 2025

@author: Sai Gunaranjan
"""


"""
LSTMs

05/08/2025
I have successfully implemented the Long Short Term Memory (LSTM) network. I have followed the following
online videos for understanding and implementing LSTMs:
    1. Andrej's lecture on LSTMs: https://www.youtube.com/watch?v=yCC09vCHzF8
    2. Stanford Lecture on LSTM: https://www.youtube.com/watch?v=6niqTuYFZLQ&t=8s
    3. Intuitive explanation of LSTMs. Why they are called Long Short Term Memory Networks: https://www.youtube.com/watch?v=YCzL96nL7j0

I have thoroughly understood and implemented the backpropagation through time for LSTMs.
I will also document my derivation of the backprop for LSTMs in OneNote.
Both RNNs and LSTMs are used to cater to classification and prediction tasks of sequential data.
In RNNs, the gradients are backpropagated through each time step via successive multiplications with
the same Whh matrix and changing diagonal matrix of activation function derivatives.
The major problem of RNNs is the vanishing gradients due to successive multiplications
of the same Whh matrix(with a diagonal matrix of activation function derivatives)
over several time steps (for a given RNN layer). The activation function in RNN is tanh and the derivative of
tanh <= 1. Now, If the eigen values of the Whh matrix are all less than one,
then after several multiplications, the output matrix contains very small values and this leads to
vanishing gradients. I will also attach a small script to illustrate this behaviour.
Now when gradients vanish, the parameter/weights do not get an update and hence the contribution from a
longer context character/word (in the context of language modelling) gets lost.
Hence, an RNN in general, cannot handle longer context windows because
the gradients diminish by the time it propagates to a far off context word or character.
This problem of vanishing gradients in RNNs is handled by making some modifications to the RNN
architecture and this leads to a new architecture called LSTMs or Long Short Term Memory Networks.
LSTMs solve the following problems of RNNs:
    1. Vanishing gradients
    2. Smaller context window for predictions
Note: LSTMs do not solve the exploding gradeints problem! So this issue still needs to be handled with some clipping or normalization!

LSTMs solve the vanishing gradeints problem by getting rid of the successive Whh matrix multiplications.
Instead it provides highway paths for the cell state (which only includes point wise multipliation and additon in its path)
through which the gradients can flow backwards without any diminishing effect.

LSTMS have 4 gates namely:
    1. Input gate,
    2. Forget gate,
    3. Output gate,
    4. G gate (which I will refer to as Gate gate!)

and 2 states(unlike RNN which has only 1 state i.e hidden state):
    1. Hidden state
    2. Cell state

The cell state can be interpreted as a Long Term Memory and the hidden state can be interpreted as a
Sort Term Memory and hence Long Short Term Memory networks!
The gates, as their names suggest, control what part of the information to recollect, forget and write down
to memory states i.e cell and hidden states.

Since, RNNs have a very small conect window, when we have large time steps to unroll and predict the output,
the RNN might forget the context of farther time steps away. But LSTMs through their gates learn
what part of the farther context information to remember, forget, and write down.
Now to learn something, mathematically, we need to introduce parameters. So we introduce weight matrices
for each of the gates.

The following are the LSTM equations for each layer:
    @ --> Matrix multiplication, * --> hadamard product or point wise multiplication
    ita_inputgate = Whh_inputgate @ PreviousHiddenState + Wxh_inputgate @ InputCurrentTimeStep
    ita_forgetgate = Whh_forgetgate @ PreviousHiddenState + Wxh_forgetgate @ InputCurrentTimeStep
    ita_outputgate = Whh_outputgate @ PreviousHiddenState + Wxh_outputgate @ InputCurrentTimeStep
    ita_gategate = Whh_gategate @ PreviousHiddenState + Wxh_gategate @ InputCurrentTimeStep

    inputgate = sigmoid(ita_inputgate)
    forgetgate = sigmoid(ita_forgetgate)
    outputgate = sigmoid(ita_outputgate)
    gategate = tanh(ita_gategate)

    currentCellState = (forgetgate * previousCellState) + (inputgate * gategate)
    currentHiddenState = outputgate * tanh(currentCellState)

    PreviousHiddenState = currentHiddenState
    previousCellState = currentCellState

Hidden state vector, cell state vector and all the gates are of the same size for a layer.
The output hidden state of current timestep i.e currentHiddenState feeds into:
    1. Next step time for the same layer as the hidden state
    2. Next layer as input for the same time step.

1. The forget gate selectively picks some part of the information from the previous cell state (long term memory).
2. The input gate selectively picks part of the information from
gategate (which is like a potential long term memory since this gets added to the cell state path or long term memory path)
1 and 2 are added to get the new/updated cell state
The updated cell state then gets squished through a tanh activation(this is like potential short term memory since it is going into the hidden state path/ short term memory path)
and the output gate selectively picks information from this to yield the new/updated hidden state
This intuitive undersatnding is well explained in StatsQuest lecture (3rd link above).

The backpropgation is a lot more involved due to these operations but I have successfully implemented it!
But, I will need to verify the correctness of the implementation. Also, I have borrowed the full connected layer
(that comes after the LSTM layers) from my DNN library implementation. There are more optimizations to be done
which I will do next. The following are the tasks:
    1. Convert explicit for loops in lists to list comprehension both for RNN as well as LSTM.
    2. Check with multi layer FC layer post NN/LST layers. Check if the code runs.
    3. Replace einsum with normal matrix multipcation(@), since it is much slower than @.[Done]
    4. Remove redundant functions which are available in both LSTM and DNN class.[Done]
    5. Replace cellStateValid with cellStateCurrent.[Done]
    6. Combine einsum and sum across time axis in the weight update step.[Not required right now]
    7. Optimize the function update_weights and compute_gradients in the DNN code.[Done]
    8. Change everywhere from RNN to lstm.[Done]
    9. LSTM Suffering from exploding gradients and code is crashing!


"""
import sys
import os

# Get parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add to sys.path
sys.path.insert(0, parent_dir)


import numpy as np
from neural_network import MLFFNeuralNetwork
import time as time
np.random.seed(1)
from textfile_preprocessing import prepare_data, get_batch
# np.seterr(over='raise')



class LSTM():

    def __init__(self, inputShape, hiddenStateVecLengthEachLSTMLayer, denseLayer, outputLayer, numTimeSteps):

        # We will assume output shape to be same as input shape, since both input and output are characters or words
        self.inputShape = inputShape
        self.numLSTMLayers = len(hiddenStateVecLengthEachLSTMLayer)
        self.outputShape = outputLayer[0][0]
        self.numTimeSteps = numTimeSteps
        # self.numTotalLayers = 1 + self.numLSTMLayers + 1 # (1 for input, 1 for output, numRNNLayers)

        """ Weight initialization"""
        """ Xavier method of weight initialization for tanh activation for vanilla RNN"""

        self.hiddenShape = hiddenStateVecLengthEachLSTMLayer # # cell state vector has the same dimension hidden state vector
        self.inputShapeEachLayer = np.zeros((self.numLSTMLayers+1,),dtype=np.int32) # Size of input to each RNN/LSTM layer and also includes the final output of the rnn/lstm layer
        self.inputShapeEachLayer[0] = self.inputShape
        self.inputShapeEachLayer[1::] = self.hiddenShape

        fanInWxh = self.inputShape+1 # inputShape+1 absorbs the bias term to the Wxh matrix

        self.WxhInputGate = []
        self.WhhInputGate = []

        self.WxhForgetGate = []
        self.WhhForgetGate = []

        self.WxhOutputGate = []
        self.WhhOutputGate = []

        self.WxhGateGate = []
        self.WhhGateGate = []

        for ele in range(self.numLSTMLayers):
            fanInWhh = fanOutWhh = self.hiddenShape[ele]
            fanOutWxh = fanOutWhh
            scalingFactorXavierInitWhh = np.sqrt(2/(fanInWhh+fanOutWhh)) * 5/3 #0.01#
            scalingFactorXavierInitWxh = np.sqrt(2/(fanInWxh+fanOutWxh)) * 5/3 #0.01#

            WhhInputGate = np.random.randn(fanOutWhh, fanInWhh) * scalingFactorXavierInitWhh
            WhhForgetGate = np.random.randn(fanOutWhh, fanInWhh) * scalingFactorXavierInitWhh
            WhhOutputGate = np.random.randn(fanOutWhh, fanInWhh) * scalingFactorXavierInitWhh
            WhhGateGate = np.random.randn(fanOutWhh, fanInWhh) * scalingFactorXavierInitWhh

            WxhInputGate = np.random.randn(fanOutWxh, fanInWxh) * scalingFactorXavierInitWxh # self.inputShape+1 absorbs the bias term to the Wxh matrix, self.numLSTMLayers+1 is for the final output layer
            WxhForgetGate = np.random.randn(fanOutWxh, fanInWxh) * scalingFactorXavierInitWxh # self.inputShape+1 absorbs the bias term to the Wxh matrix, self.numLSTMLayers+1 is for the final output layer
            WxhOutputGate = np.random.randn(fanOutWxh, fanInWxh) * scalingFactorXavierInitWxh # self.inputShape+1 absorbs the bias term to the Wxh matrix, self.numLSTMLayers+1 is for the final output layer
            WxhGateGate = np.random.randn(fanOutWxh, fanInWxh) * scalingFactorXavierInitWxh # self.inputShape+1 absorbs the bias term to the Wxh matrix, self.numLSTMLayers+1 is for the final output layer

            fanInWxh = fanOutWhh + 1 # fanOutWhh+1 absorbs the bias term to the Wxh matrix

            self.WxhInputGate.append(WxhInputGate)
            self.WxhForgetGate.append(WxhForgetGate)
            self.WxhOutputGate.append(WxhOutputGate)
            self.WxhGateGate.append(WxhGateGate)


            self.WhhInputGate.append(WhhInputGate)
            self.WhhForgetGate.append(WhhForgetGate)
            self.WhhOutputGate.append(WhhOutputGate)
            self.WhhGateGate.append(WhhGateGate)

        # Define the dense layer/FC layer
        lstmOutputLayer = [(self.hiddenShape[-1],'Identity',0)] # input to dense layer will not have BN
        denseLayerArchitecture = lstmOutputLayer + denseLayer + outputLayer
        self.mlffnn = MLFFNeuralNetwork(denseLayerArchitecture)


        # print('Im here')



    def preprocess_textfile(self,textfile):

        self.params = prepare_data(textfile, n_segments = self.mlffnn.batchsize, seq_len = self.numTimeSteps)



    def define_hiddenstatematrix(self,hiddenState):
        self.hiddenStateMatrix = []
        for ele1 in range(self.numLSTMLayers):
        	temp = np.zeros((self.numTimeSteps+1, self.hiddenShape[ele1], self.mlffnn.batchsize),dtype=np.float32)
        	temp[0,:,:] = hiddenState[ele1] # Write the hidden state at the final time step back to the inital time step for next batch
        	self.hiddenStateMatrix.append(temp)


    def define_cellstatematrix(self,cellState):
        self.cellStateMatrix = []
        for ele1 in range(self.numLSTMLayers):
        	temp = np.zeros((self.numTimeSteps+1, self.hiddenShape[ele1], self.mlffnn.batchsize),dtype=np.float32)
        	temp[0,:,:] = cellState[ele1] # Write the hidden state at the final time step back to the inital time step for next batch
        	self.cellStateMatrix.append(temp)



    def define_gatematrix(self):
        # Used to store values of the gates to be later on used in backward pass
        self.inputGate = []
        self.forgetGate = []
        self.outputGate = []
        self.gateGate = []
        for ele1 in range(self.numLSTMLayers):
            cell = np.zeros((self.numTimeSteps, self.hiddenShape[ele1], self.mlffnn.batchsize),dtype=np.float32)
            self.inputGate.append(cell)
            self.forgetGate.append(cell)
            self.outputGate.append(cell)
            self.gateGate.append(cell)



    def define_inputmatrix(self,trainDataSample):
        self.inputMatrixX = []
        for layer in range(self.numLSTMLayers+1):
        	cell = np.ones((self.numTimeSteps, self.inputShapeEachLayer[layer]+1, self.mlffnn.batchsize), dtype=np.float32) ## self.inputShape+1 is for the bias
        	if layer == 0:
        		cell[:,1::, :] = trainDataSample # 1st term is for the bias term
        	self.inputMatrixX.append(cell)


    def define_outputmatrix(self):
        self.outputMatrix = []
        for layer in range(self.numLSTMLayers):
        	cell = np.zeros((self.numTimeSteps, self.hiddenShape[layer], self.mlffnn.batchsize), dtype=np.float32)
        	self.outputMatrix.append(cell)


    def define_itamatrix(self):

        self.itaInputGate = [row.copy() for row in self.outputMatrix] # ita is same size as output matrix
        self.itaForgetGate = [row.copy() for row in self.outputMatrix] # ita is same size as output matrix
        self.itaOutputGate = [row.copy() for row in self.outputMatrix] # ita is same size as output matrix
        self.itaGateGate = [row.copy() for row in self.outputMatrix] # ita is same size as output matrix


    def forwardpass_lstm(self, trainDataSample, trainDataLabel, hiddenState, cellState):

        # numTrainingSamples = trainDataSample.shape[2] # batch size, 1st dimension is sequence length, 2nd dimension is length of vector, 3rd dimension is batch size

        self.define_hiddenstatematrix(hiddenState)
        self.define_cellstatematrix(cellState)
        self.define_gatematrix()
        self.define_inputmatrix(trainDataSample)
        self.define_outputmatrix()
        self.define_itamatrix()

        self.errorGradFeedIntoLSTM = [] # These are the error gradients feeding into the LSTM from the DNN after backprop in the MLFFNN for each time step
        self.dnnWeightGradients = []
        self.dnnBatchNormGammaScalingGradients = []
        self.dnnBatchNormBetaShiftGradients = []
        self.predictedOutputAllTimeSteps = np.zeros((self.numTimeSteps,self.outputShape,self.mlffnn.batchsize),dtype=np.float32)
        for ele2 in range(self.numTimeSteps):

            for ele1 in range(self.numLSTMLayers):

                hiddenStateTminus1LayerLplus1 = self.hiddenStateMatrix[ele1][ele2,:,:]
                cellStateTminus1 = self.cellStateMatrix[ele1][ele2,:,:]
                hiddenStateTLayerL = self.inputMatrixX[ele1][ele2,:,:]

                itaLayerLPlus1InputGate = (self.WhhInputGate[ele1] @ hiddenStateTminus1LayerLplus1) + (self.WxhInputGate[ele1] @ hiddenStateTLayerL)
                itaLayerLPlus1ForgetGate = (self.WhhForgetGate[ele1] @ hiddenStateTminus1LayerLplus1) + (self.WxhForgetGate[ele1] @ hiddenStateTLayerL)
                itaLayerLPlus1OutputGate = (self.WhhOutputGate[ele1] @ hiddenStateTminus1LayerLplus1) + (self.WxhOutputGate[ele1] @ hiddenStateTLayerL)
                itaLayerLPlus1GateGate = (self.WhhGateGate[ele1] @ hiddenStateTminus1LayerLplus1) + (self.WxhGateGate[ele1] @ hiddenStateTLayerL)

                self.itaInputGate[ele1][ele2,:,:] = itaLayerLPlus1InputGate
                self.itaForgetGate[ele1][ele2,:,:] = itaLayerLPlus1ForgetGate
                self.itaOutputGate[ele1][ele2,:,:] = itaLayerLPlus1OutputGate
                self.itaGateGate[ele1][ele2,:,:] = itaLayerLPlus1GateGate

                inputGate = self.mlffnn.activation_function(itaLayerLPlus1InputGate, 'sigmoid')
                forgetGate = self.mlffnn.activation_function(itaLayerLPlus1ForgetGate, 'sigmoid')
                outputGate = self.mlffnn.activation_function(itaLayerLPlus1OutputGate, 'sigmoid')
                gateGate = self.mlffnn.activation_function(itaLayerLPlus1GateGate, 'tanh')

                cellStateT = (forgetGate * cellStateTminus1) + (inputGate * gateGate )
                hiddenStateTLayerLplus1 = outputGate * self.mlffnn.activation_function(cellStateT, 'tanh')

                self.hiddenStateMatrix[ele1][ele2+1,:,:] = hiddenStateTLayerLplus1
                self.cellStateMatrix[ele1][ele2+1,:,:] = cellStateT

                self.inputGate[ele1][ele2,:,:] = inputGate
                self.forgetGate[ele1][ele2,:,:] = forgetGate
                self.outputGate[ele1][ele2,:,:] = outputGate
                self.gateGate[ele1][ele2,:,:] = gateGate


                self.inputMatrixX[ele1+1][ele2,1::,:] = hiddenStateTLayerLplus1
                self.outputMatrix[ele1][ele2,:,:] = hiddenStateTLayerLplus1

            """ For every time step, for the DNN part, do the forward pass, backward pass,
            compute and store the weight gradients and errors feeding into the LSTM"""
            self.mlffnn.forwardpass(hiddenStateTLayerLplus1, 'train')
            self.predictedOutputAllTimeSteps[ele2,:,:] = self.mlffnn.predictedOutput # Store final output for each time step
            self.mlffnn.backwardpass(trainDataLabel[ele2,:,:])
            """ Back propagating error from dense layer to last RNN/LSTM layer"""
            doLbydoItaDNN = self.mlffnn.errorEachLayer[-1]
            self.errorGradFeedIntoLSTM.append(doLbydoItaDNN) #doLbydoItaDNN
            self.mlffnn.compute_gradients() # Compute do L / do W for each layer of DNN
            self.dnnWeightGradients.append(self.mlffnn.gradientCostFnwrtWeights) # Store the weight gradients of DNN for each time step. This will be later on summed across time steps to get the overall gradeients
            self.dnnBatchNormGammaScalingGradients.append(self.mlffnn.gradientCostFnwrtGammaScaling) # If BN is enabled, these gradients also have to be computed
            self.dnnBatchNormBetaShiftGradients.append(self.mlffnn.gradientCostFnwrtBetaShift)


        hiddenState = [row[-1,:,:] for row in self.hiddenStateMatrix] # hidden state of each layer and last time step is stored and used as input hidden state of next charcter
        cellState = [row[-1,:,:] for row in self.cellStateMatrix] # cell state of each layer and last time step is stored and used as input hidden state of next charcter

        self.cellStateCurrent = [row[1::,:,:] for row in self.cellStateMatrix]
        self.cellStatePrevious = [row[0:-1,:,:] for row in self.cellStateMatrix]

        return hiddenState, cellState


    def backwardpass_lstm(self,trainDataLabel):
        # trainDataLabel should be of shape numTimeSteps, outputShape, batch size

        #Errors/delta = dL/d ht, dL/d ct
        self.doLbydoht = [np.zeros(row.shape, dtype=row.dtype) for row in self.outputMatrix]
        self.doLbydoct = [np.zeros(row.shape, dtype=row.dtype) for row in self.outputMatrix]

        for ele2 in range(self.numTimeSteps-1,-1,-1):

            """ Back propagating error from dense layer to last RNN/LSTM layer"""
            doLbydoItaDNN = self.errorGradFeedIntoLSTM[ele2]
            weightMatrixLayer1to2 = self.mlffnn.weightMatrixList[0]
            self.doLbydoht[-1][ele2,:,:] = (weightMatrixLayer1to2[:,1::].T @ doLbydoItaDNN)

            for ele1 in range(self.numLSTMLayers-1,-1,-1): #range(self.numLSTMLayers-1,-1,-1):

                if ((ele1 == self.numLSTMLayers-1) and (ele2 == self.numTimeSteps-1)):
                    # Don't update self.doLbydoht[ele1][ele2,:,:]. Just compute self.doLbydoct[ele1][ele2,:,:]
                    # as a scaled version of self.doLbydoht[ele1][ele2,:,:]
                    # Last RNN/LSTM layer, last time step condition to define dolbydoct

                    gradctFromht = self.gradient_ct_from_ht(ele1,ele2)
                    self.doLbydoct[ele1][ele2,:,:] = gradctFromht


                if ((ele1 == self.numLSTMLayers-1) and (ele2 != self.numTimeSteps-1)):
                    # update self.doLbydoht[ele1][ele2,:,:] with contribution from Whh ele2+1.
                    # Also compute self.doLbydoct[ele1][ele2,:,:] with contribition from ele2+1

                    gradFromLayerLTimeTplus1 = self.gradient_ht_from_right(ele1,ele2)
                    self.doLbydoht[ele1][ele2,:,:] += gradFromLayerLTimeTplus1

                    # Update cell state
                    gradctFromctPlus1 = self.gradient_ct_from_ctplus1(ele1,ele2)

                    gradctFromht = self.gradient_ct_from_ht(ele1,ele2)

                    self.doLbydoct[ele1][ele2,:,:] =  gradctFromctPlus1 + gradctFromht



                if ((ele1 != self.numLSTMLayers-1) and (ele2 == self.numTimeSteps-1)):
                    # compute self.doLbydoht[ele1][ele2,:,:] with contribition from ele1+1.
                    # Also, compute self.doLbydoct[ele1][ele2,:,:] as a scaled version of self.doLbydoht[ele1][ele2,:,:]

                    gradFromLayerLplus1TimeT = self.gradient_ht_from_top(ele1,ele2)
                    self.doLbydoht[ele1][ele2,:,:] = gradFromLayerLplus1TimeT

                    # compute doLbydoct
                    gradctFromht = self.gradient_ct_from_ht(ele1,ele2)
                    self.doLbydoct[ele1][ele2,:,:] = gradctFromht



                if ((ele1 != self.numLSTMLayers-1) and (ele2 != self.numTimeSteps-1)):
                    # compute self.doLbydoht[ele1][ele2,:,:] with contribition from top layer and next time step.
                    # Also, compute self.doLbydoct[ele1][ele2,:,:] from next time step of self.doLbydoct[ele1][ele2,:,:]

                    # Contribution from top layer error
                    gradFromLayerLplus1TimeT = self.gradient_ht_from_top(ele1,ele2)

                    ## Gradient from t+1
                    gradFromLayerLTimeTplus1 = self.gradient_ht_from_right(ele1,ele2)


                    self.doLbydoht[ele1][ele2,:,:] = gradFromLayerLplus1TimeT + gradFromLayerLTimeTplus1

                    # Update cell state

                    gradctFromctPlus1 = self.gradient_ct_from_ctplus1(ele1,ele2)

                    gradctFromht = self.gradient_ct_from_ht(ele1,ele2)

                    self.doLbydoct[ele1][ele2,:,:] =  gradctFromctPlus1 + gradctFromht




        # Need to change below lines for 2d lists of 2d arrays
        # np.clip(self.errorMatrix, -5, 5, out=self.errorMatrix) # Clip to prevent exploding gradients

        # plt.hist(self.errorMatrix[0][:,:,0].flatten(),bins=50)
        # print('\n\n')
        # print('Min value of gradient: {0:.1f}'.format(np.amin(self.errorMatrix[0][:,:,0].flatten())))
        # print('Max value of gradient: {0:.1f}'.format(np.amax(self.errorMatrix[0][:,:,0].flatten())))
        # percentile = 95
        # print('{0} percentile value of gradient: {1:.1f}'.format(percentile, np.percentile(np.abs(self.errorMatrix[0][:,:,0].flatten()),percentile)))

        # print('Std at last time step = {0:.3f}'.format(np.std(self.errorMatrix[0][-1,:,0])))
        # print('Std at first time step = {0:.3f}'.format(np.std(self.errorMatrix[0][0,:,0])))

        # print('--')

    def gradient_ct_from_ht(self,ele1,ele2):

        outputGateT = self.outputGate[ele1][ele2,:,:]
        cellStateT = self.cellStateCurrent[ele1][ele2,:,:] # check the indexing
        activationFnDerivativecellState = self.mlffnn.derivative_activation_function(cellStateT,'tanh')
        gradctFromht = self.doLbydoht[ele1][ele2,:,:] * activationFnDerivativecellState * outputGateT

        return gradctFromht


    def gradient_ct_from_ctplus1(self,ele1,ele2):

        forgetGateTplus1 = self.forgetGate[ele1][ele2+1,:,:]
        gradctFromctPlus1 = (self.doLbydoct[ele1][ele2+1,:,:] * forgetGateTplus1)

        return gradctFromctPlus1



    def gradient_ht_from_top(self,ele1,ele2):

        del_ht_lplus1 = self.doLbydoht[ele1+1][ele2,:,:]

        # del_ht_lplus1 has to pass through the contribution from each of the gates

        """ Input gate contribution"""
        itaInputGateLayerL = self.itaInputGate[ele1+1][ele2,:,:]
        activationFnDerivativeInputGate = self.mlffnn.derivative_activation_function(itaInputGateLayerL,'sigmoid')
        gateGateT = self.gateGate[ele1+1][ele2,:,:]
        outputGateT = self.outputGate[ele1+1][ele2,:,:]
        cellStateT = self.cellStateCurrent[ele1+1][ele2,:,:] # check the indexing
        activationFnDerivativecellState = self.mlffnn.derivative_activation_function(cellStateT,'tanh')
        inputGatePath = activationFnDerivativeInputGate * gateGateT * outputGateT * \
                activationFnDerivativecellState * del_ht_lplus1


        """ Forget gate contribution"""
        itaForgetGateLayerL = self.itaForgetGate[ele1+1][ele2,:,:]
        activationFnDerivativeForgetGate = self.mlffnn.derivative_activation_function(itaForgetGateLayerL,'sigmoid')
        cellStateTminus1 = self.cellStateCurrent[ele1+1][ele2-1,:,:] # check the indexing
        outputGateT = self.outputGate[ele1+1][ele2,:,:]
        cellStateT = self.cellStateCurrent[ele1+1][ele2,:,:] # check the indexing
        activationFnDerivativecellState = self.mlffnn.derivative_activation_function(cellStateT,'tanh')
        forgetGatePath = activationFnDerivativeForgetGate * \
            cellStateTminus1 * outputGateT * \
                activationFnDerivativecellState * del_ht_lplus1


        """ Output gate contribution"""
        itaOutputGateLayerL = self.itaOutputGate[ele1+1][ele2,:,:]
        activationFnDerivativeOutputGate = self.mlffnn.derivative_activation_function(itaOutputGateLayerL,'sigmoid')
        cellStateT = self.cellStateCurrent[ele1+1][ele2,:,:] # check the indexing
        activationFnDerivativecellState = self.mlffnn.derivative_activation_function(cellStateT,'tanh')
        outputGatePath = activationFnDerivativeOutputGate * \
                activationFnDerivativecellState * del_ht_lplus1


        """ Gate gate contribution"""
        itaGateGateLayerL = self.itaGateGate[ele1+1][ele2,:,:]
        activationFnDerivativeGateGate = self.mlffnn.derivative_activation_function(itaGateGateLayerL,'tanh')
        inputGateT = self.inputGate[ele1+1][ele2,:,:]
        outputGateT = self.outputGate[ele1+1][ele2,:,:]
        cellStateT = self.cellStateCurrent[ele1+1][ele2,:,:] # check the indexing
        activationFnDerivativecellState = self.mlffnn.derivative_activation_function(cellStateT,'tanh')
        gateGatePath = activationFnDerivativeGateGate * \
            inputGateT * outputGateT * \
                activationFnDerivativecellState * del_ht_lplus1


        gradFromLayerLplus1TimeT = (self.WxhInputGate[ele1+1][:,1::].T @ inputGatePath) + \
           (self.WxhForgetGate[ele1+1][:,1::].T @ forgetGatePath) + \
               (self.WxhOutputGate[ele1+1][:,1::].T @ outputGatePath) + \
                   (self.WxhGateGate[ele1+1][:,1::].T @ gateGatePath)

        return gradFromLayerLplus1TimeT



    def gradient_ht_from_right(self,ele1,ele2):

        del_htplus1_l = self.doLbydoht[ele1][ele2+1,:,:]


        """ Input gate contribution"""
        itaInputGateLayerL = self.itaInputGate[ele1][ele2+1,:,:]
        activationFnDerivativeInputGate = self.mlffnn.derivative_activation_function(itaInputGateLayerL,'sigmoid')
        gateGateT = self.gateGate[ele1][ele2+1,:,:]
        outputGateT = self.outputGate[ele1][ele2+1,:,:]
        cellStateT = self.cellStateCurrent[ele1][ele2+1,:,:] # check the indexing
        activationFnDerivativecellState = self.mlffnn.derivative_activation_function(cellStateT,'tanh')
        inputGatePath = activationFnDerivativeInputGate * gateGateT * outputGateT * \
                activationFnDerivativecellState * del_htplus1_l


        """ Forget gate contribution"""
        itaForgetGateLayerL = self.itaForgetGate[ele1][ele2+1,:,:]
        activationFnDerivativeForgetGate = self.mlffnn.derivative_activation_function(itaForgetGateLayerL,'sigmoid')
        cellStateTminus1 = self.cellStateCurrent[ele1][ele2,:,:] # check the indexing
        outputGateT = self.outputGate[ele1][ele2+1,:,:]
        cellStateT = self.cellStateCurrent[ele1][ele2+1,:,:] # check the indexing
        activationFnDerivativecellState = self.mlffnn.derivative_activation_function(cellStateT,'tanh')
        forgetGatePath = activationFnDerivativeForgetGate * \
            cellStateTminus1 * outputGateT * \
                activationFnDerivativecellState * del_htplus1_l


        """ Output gate contribution"""
        itaOutputGateLayerL = self.itaOutputGate[ele1][ele2+1,:,:]
        activationFnDerivativeOutputGate = self.mlffnn.derivative_activation_function(itaOutputGateLayerL,'sigmoid')
        cellStateT = self.cellStateCurrent[ele1][ele2+1,:,:] # check the indexing
        activationFnDerivativecellState = self.mlffnn.derivative_activation_function(cellStateT,'tanh')
        outputGatePath = activationFnDerivativeOutputGate * \
                activationFnDerivativecellState * del_htplus1_l


        """ Gate gate contribution"""
        itaGateGateLayerL = self.itaGateGate[ele1][ele2+1,:,:]
        activationFnDerivativeGateGate = self.mlffnn.derivative_activation_function(itaGateGateLayerL,'tanh')
        inputGateT = self.inputGate[ele1][ele2+1,:,:]
        outputGateT = self.outputGate[ele1][ele2+1,:,:]
        cellStateT = self.cellStateMatrix[ele1][ele2+1,:,:] # check the indexing
        activationFnDerivativecellState = self.mlffnn.derivative_activation_function(cellStateT,'tanh')
        gateGatePath = activationFnDerivativeGateGate * \
            inputGateT * outputGateT * \
                activationFnDerivativecellState * del_htplus1_l


        gradFromLayerLTimeTplus1 = ((self.WhhInputGate[ele1].T @ inputGatePath) + \
            (self.WhhForgetGate[ele1].T @ forgetGatePath) + \
                (self.WhhOutputGate[ele1].T @ outputGatePath) + \
                    (self.WhhGateGate[ele1].T @ gateGatePath))

        return gradFromLayerLTimeTplus1



    def gradient_wrt_inputgate_weightmatrix(self,ele1,outputEachLayer):

        itaInputGateLayerL = self.itaInputGate[ele1]
        doItbydoItaInputGate = self.mlffnn.derivative_activation_function(itaInputGateLayerL,'sigmoid')
        doCtbydoIt = self.gateGate[ele1]
        doLbydoItaInputGate = self.doLbydoct[ele1] * doCtbydoIt * doItbydoItaInputGate
        tempGradientsWInputGate = np.einsum('ijk,ilk->ilj',outputEachLayer[ele1], doLbydoItaInputGate,optimize=True)/self.mlffnn.batchsize
        gradientCostFnwrtWInputGate = np.sum(tempGradientsWInputGate,axis=0) # Sum across time steps

        return gradientCostFnwrtWInputGate


    def gradient_wrt_forgetgate_weightmatrix(self,ele1,outputEachLayer):

        itaForgetGateLayerL = self.itaForgetGate[ele1]
        doFtbydoItaForgetGate = self.mlffnn.derivative_activation_function(itaForgetGateLayerL,'sigmoid')
        doCtbydoFt = self.cellStatePrevious[ele1] # Read as do ct / do ft = ct-1
        doLbydoItaForgetGate = self.doLbydoct[ele1] * doCtbydoFt * doFtbydoItaForgetGate
        tempGradientsWForgetGate = np.einsum('ijk,ilk->ilj',outputEachLayer[ele1], doLbydoItaForgetGate,optimize=True)/self.mlffnn.batchsize
        gradientCostFnwrtWForgetGate = np.sum(tempGradientsWForgetGate,axis=0) # Sum across time steps

        return gradientCostFnwrtWForgetGate


    def gradient_wrt_outputgate_weightmatrix(self,ele1,outputEachLayer):

        itaOutputGateLayerL = self.itaOutputGate[ele1]
        doOtbydoItaOutputGate = self.mlffnn.derivative_activation_function(itaOutputGateLayerL,'sigmoid')
        doHtbydoOt = self.mlffnn.activation_function(self.cellStateCurrent[ele1],'tanh')
        doLbydoItaOutputGate = self.doLbydoht[ele1] * doHtbydoOt * doOtbydoItaOutputGate
        tempGradientsWOutputGate = np.einsum('ijk,ilk->ilj',outputEachLayer[ele1], doLbydoItaOutputGate,optimize=True)/self.mlffnn.batchsize
        gradientCostFnwrtWOutputGate = np.sum(tempGradientsWOutputGate,axis=0) # Sum across time steps

        return gradientCostFnwrtWOutputGate


    def gradient_wrt_gategate_weightmatrix(self,ele1,outputEachLayer):

        itaGateGateLayerL = self.itaGateGate[ele1]
        doGtbydoItaGateGate = self.mlffnn.derivative_activation_function(itaGateGateLayerL,'tanh')
        doCtbydoGt = self.inputGate[ele1]
        doLbydoItaOutputGate = self.doLbydoct[ele1] * doCtbydoGt * doGtbydoItaGateGate
        tempGradientsWGateGate = np.einsum('ijk,ilk->ilj',outputEachLayer[ele1], doLbydoItaOutputGate,optimize=True)/self.mlffnn.batchsize
        gradientCostFnwrtWGateGate = np.sum(tempGradientsWGateGate,axis=0) # Sum across time steps

        return gradientCostFnwrtWGateGate


    def update_weights_lstm(self):


        # 1. Slice layers from inputMatrixX (exclude last layer)
        outputEachLayer = [row.copy() for row in self.inputMatrixX[0:self.numLSTMLayers]]
        # 2. Slice hiddenStateMatrix for all layers and only first numTimeSteps
        hiddenStateEachLayer = [row[0:self.numTimeSteps,:,:].copy() for row in self.hiddenStateMatrix]


        for ele1 in range(self.numLSTMLayers):

            """ do L / do Wxh """

            gradientCostFnwrtWxhInputGate = self.gradient_wrt_inputgate_weightmatrix(ele1,outputEachLayer)
            self.WxhInputGate[ele1] -= self.mlffnn.stepsize*gradientCostFnwrtWxhInputGate

            gradientCostFnwrtWxhForgetGate = self.gradient_wrt_forgetgate_weightmatrix(ele1,outputEachLayer)
            self.WxhForgetGate[ele1] -= self.mlffnn.stepsize*gradientCostFnwrtWxhForgetGate


            gradientCostFnwrtWxhOutputGate = self.gradient_wrt_outputgate_weightmatrix(ele1,outputEachLayer)
            self.WxhOutputGate[ele1] -= self.mlffnn.stepsize*gradientCostFnwrtWxhOutputGate

            gradientCostFnwrtWxhGateGate = self.gradient_wrt_gategate_weightmatrix(ele1,outputEachLayer)
            self.WxhGateGate[ele1] -= self.mlffnn.stepsize*gradientCostFnwrtWxhGateGate


            """ do L / do Whh """

            gradientCostFnwrtWhhInputGate = self.gradient_wrt_inputgate_weightmatrix(ele1,hiddenStateEachLayer)
            self.WhhInputGate[ele1] -= self.mlffnn.stepsize*gradientCostFnwrtWhhInputGate

            gradientCostFnwrtWhhForgetGate = self.gradient_wrt_forgetgate_weightmatrix(ele1,hiddenStateEachLayer)
            self.WhhForgetGate[ele1] -= self.mlffnn.stepsize*gradientCostFnwrtWhhForgetGate


            gradientCostFnwrtWhhOutputGate = self.gradient_wrt_outputgate_weightmatrix(ele1,hiddenStateEachLayer)
            self.WhhOutputGate[ele1] -= self.mlffnn.stepsize*gradientCostFnwrtWhhOutputGate

            gradientCostFnwrtWhhGateGate = self.gradient_wrt_gategate_weightmatrix(ele1,hiddenStateEachLayer)
            self.WhhGateGate[ele1] -= self.mlffnn.stepsize*gradientCostFnwrtWhhGateGate


            # Below line needs to be modified for lists
            # np.clip(gradientCostFnwrtWxh, -5, 5, out=gradientCostFnwrtWxh)


            # Below line needs to be modified for lists
            # np.clip(gradientCostFnwrtWhh, -5, 5, out=gradientCostFnwrtWhh)

        """ Take sum of all DNN parameter gradients. This includes weights, BN parameters"""
        gradientCostFnwrtDNNWeights = [sum(arrays) for arrays in zip(*self.dnnWeightGradients)]
        gradientCostFnwrtDNNBNGammaScaling = [sum(arrays) for arrays in zip(*self.dnnBatchNormGammaScalingGradients)]
        gradientCostFnwrtDNNBNBetaShift = [sum(arrays) for arrays in zip(*self.dnnBatchNormBetaShiftGradients)]
        """ Updates the parameters of the DNN like weights, BN parameters using gradient descent"""
        self.mlffnn.update_weights(gradientCostFnwrtDNNWeights, gradientCostFnwrtDNNBNGammaScaling, gradientCostFnwrtDNNBNBetaShift)

        # print('Im here')







    def compute_forward_backward_pass_lstm(self, trainDataSample, trainDataLabel, hiddenState, cellState):

        """ Forward pass"""
        t1 = time.time()
        hiddenState, cellState = self.forwardpass_lstm(trainDataSample, trainDataLabel, hiddenState, cellState)
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
        self.backwardpass_lstm(trainDataLabel)
        t6 = time.time()
        # print('Time taken for backward pass = {0:.2f} s'.format(t6-t5))

        """ Update weights"""
        t7 = time.time()
        self.update_weights_lstm()
        t8 = time.time()
        # print('Time taken for Weight update = {0:.2f} s'.format(t8-t7))

        return hiddenState, cellState


    def backpropagation_lstm(self):

        flagStepSizeChange = 1
        self.trainingLossArray = []
        self.validationLossArray = []
        for ele1 in np.arange(self.mlffnn.epochs):
            timeEpochStart = time.time()

            self.mini_batch_gradient_descent_lstm()

            """ Training loss and accuracy post each epoch"""
            t3 = time.time()
            self.compute_train_loss_acc_lstm()
            t4 = time.time()
            # print('Time taken for computing training loss and accuracy after epoch = {0:.2f} min'.format(t4-t3)/60)

            """ There is always validation data to test model"""
            timeStartValidation = time.time()
            self.compute_validation_loss_acc_lstm()
            timeEndValidation = time.time()
            timeValidation = (timeEndValidation - timeStartValidation)

            print('\ntrain_loss: {0:.1f}, val_loss: {1:.1f}, train_accuracy: {2:.1f}, val_accuracy: {3:.1f}'.format(self.trainingLoss, self.validationLoss, self.trainAccuracy, self.validationAccuracy))
            # Add a prediction after each epoch just to check the performance
            if ((self.trainAccuracy > 80) and (self.validationAccuracy > 80) and (flagStepSizeChange == 1)): # Ideally it should be ((self.trainAccuracy > 90) and (self.validationAccuracy > 90)
                self.mlffnn.stepsize = self.mlffnn.stepsize/10 # Make step size smaller when achieving higher accuracy > 90%
                flagStepSizeChange = 0

            if ((self.trainAccuracy > 95) and (self.validationAccuracy > 95)):
                break


            timeEpochEnd = time.time()
            timeEachEpoch = (timeEpochEnd - timeEpochStart)/60
            print('Time taken for epoch {0}/{1} = {2:.2f} min'.format(ele1+1, self.mlffnn.epochs, timeEachEpoch))

            predSeqLen = 200
            self.predict(predSeqLen) # Generate a character sequence of length = predSeqLen, at the end of each epoch



    def compute_loss_function(self,trainDataLabel, predictedOutput):

        # Cost function = 'categorical_cross_entropy'
        mask = predictedOutput !=0 # Avoid 0 values in log2 evaluation. But this is not correct. It can mask wrong classifications.
        N = predictedOutput.shape[1] * predictedOutput.shape[2] # numTimeSteps * numExamples
        # cost fn = -Sum(di*log(yi))/N, where di is the actual output and yi is the predicted output, N is the batch size.
        costFunction = (-np.sum((trainDataLabel[mask]*np.log2(predictedOutput[mask]))))/N # Mean loss across data points
        """ Need to divide by N (batch size) to get the mean loss across data points"""

        return costFunction



    def train(self):

        self.backpropagation_lstm()



    def mini_batch_gradient_descent_lstm(self):

        randBatchInd = np.random.randint(0,self.params["n_train_batches"])
        """ For stateful RNN, we may not need to shuffle the data while training, I think. Will verify this!"""
        hiddenState = [np.zeros((self.hiddenShape[ele], self.mlffnn.batchsize), dtype=np.float32) for ele in range(self.numLSTMLayers)]
        cellState = [np.zeros((self.hiddenShape[ele], self.mlffnn.batchsize), dtype=np.float32) for ele in range(self.numLSTMLayers)]
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
                self.cellStateForPredict = [c[:, 0] for c in cellState] # Sample the previous cell state for 1 example since prediction works with 1 sample at a time
                self.startIdx = np.argmax(trainDataSample[0,0,:]) # Store the starting character idx for the next sequence starting
            t1 = time.time()
            trainDataSample = np.transpose(trainDataSample,(1,2,0))
            trainDataLabel = np.transpose(trainDataLabel,(1,2,0))
            hiddenState, cellState = self.compute_forward_backward_pass_lstm(trainDataSample,trainDataLabel, hiddenState, cellState)
            t2 = time.time()





    def compute_train_loss_acc_lstm(self):

        """ Compute training loss and accuracy on the training data again with the weights obtained at the end of each epoch
        Hidden state is set back to 0 when evaluating the training loss/accuracy after training for each epoch.
        But I could as well use the hidden state from the last example of the last time step of previous epoch!

        But within an epoch, the hidden state is carried forward across the batches and examples"""

        actualOutputAllTrainData = np.zeros((self.numTimeSteps,self.outputShape,self.mlffnn.batchsize, self.params["n_train_batches"]))
        predictedOutputAllTrainData = np.zeros((self.numTimeSteps,self.outputShape,self.mlffnn.batchsize, self.params["n_train_batches"]))
        hiddenState = [np.zeros((self.hiddenShape[ele], self.mlffnn.batchsize), dtype=np.float32) for ele in range(self.numLSTMLayers)] # Currently hidden state being rolled back to 0!
        cellState = [np.zeros((self.hiddenShape[ele], self.mlffnn.batchsize), dtype=np.float32) for ele in range(self.numLSTMLayers)] # Currently cell state being rolled back to 0!
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
            hiddenState, cellState = self.forwardpass_lstm(trainDataSample,trainDataLabel,hiddenState,cellState)
            predictedOutputAllTrainData[:,:,:,batch_step] = self.predictedOutputAllTimeSteps
            actualOutputAllTrainData[:,:,:,batch_step] = trainDataLabel


        predictedOutputAllTrainData = predictedOutputAllTrainData.reshape(self.numTimeSteps,self.outputShape,self.mlffnn.batchsize*self.params["n_train_batches"])
        predictedOutputAllTrainData = np.transpose(predictedOutputAllTrainData,(1,0,2))
        actualOutputAllTrainData = actualOutputAllTrainData.reshape(self.numTimeSteps,self.outputShape,self.mlffnn.batchsize*self.params["n_train_batches"])
        actualOutputAllTrainData = np.transpose(actualOutputAllTrainData,(1,0,2))
        self.trainingLoss = self.compute_loss_function(actualOutputAllTrainData, predictedOutputAllTrainData)
        self.trainingLossArray.append(self.trainingLoss) # Keep appending the cost/loss function value for each epoch
        self.mlffnn.get_accuracy(actualOutputAllTrainData, predictedOutputAllTrainData)
        self.trainAccuracy = self.mlffnn.accuracy



    def compute_validation_loss_acc_lstm(self):

        """ Compute validation loss and accuracy on the validation data with the weights obtained at the end of each epoch
        Here also, hidden state is set back to 0 when evaluating the validation loss/accuracy after training for each epoch.
        But I could as well use the hidden state from the last example of the last time step of previous epoch!

        But within an epoch, the hidden state is carried forward across the batches and examples
        """


        actualOutputAllValidationData = np.zeros((self.numTimeSteps,self.outputShape,self.mlffnn.batchsize, self.params["n_val_batches"]))
        predictedOutputAllValidationData = np.zeros((self.numTimeSteps,self.outputShape,self.mlffnn.batchsize, self.params["n_val_batches"]))
        hiddenState = [np.zeros((self.hiddenShape[ele], self.mlffnn.batchsize), dtype=np.float32) for ele in range(self.numLSTMLayers)] # Currently hidden state being rolled back to 0!
        cellState = [np.zeros((self.hiddenShape[ele], self.mlffnn.batchsize), dtype=np.float32) for ele in range(self.numLSTMLayers)] # Currently cell state being rolled back to 0!
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
            hiddenState,cellState = self.forwardpass_lstm(validationDataSample,validationDataLabel,hiddenState,cellState)
            predictedOutputAllValidationData[:,:,:,batch_step] = self.predictedOutputAllTimeSteps
            actualOutputAllValidationData[:,:,:,batch_step] = validationDataLabel


        predictedOutputAllValidationData = predictedOutputAllValidationData.reshape(self.numTimeSteps,self.outputShape,self.mlffnn.batchsize*self.params["n_val_batches"])
        predictedOutputAllValidationData = np.transpose(predictedOutputAllValidationData,(1,0,2))
        actualOutputAllValidationData = actualOutputAllValidationData.reshape(self.numTimeSteps,self.outputShape,self.mlffnn.batchsize*self.params["n_val_batches"])
        actualOutputAllValidationData = np.transpose(actualOutputAllValidationData,(1,0,2))
        self.validationLoss = self.compute_loss_function(actualOutputAllValidationData, predictedOutputAllValidationData)
        self.validationLossArray.append(self.validationLoss) # Keep appending the cost/loss function value for each epoch
        self.mlffnn.get_accuracy(actualOutputAllValidationData, predictedOutputAllValidationData)
        self.validationAccuracy = self.mlffnn.accuracy



    def predict(self, predSeqLen):


        textString = ''
        hiddenStateTminus1LayerLplus1 = [arr.copy() for arr in self.hiddenStateForPredict]
        cellStateTminus1 = [arr.copy() for arr in self.cellStateForPredict]
        idx = self.startIdx
        startingchar = self.params['idx2char'][idx]
        textString += startingchar
        inputVector = np.zeros((self.inputShape+1,)) # +1 for the bias
        inputVector[0] = 1 # 1st element is for the bias term
        inputVector[idx+1] = 1 # To account for the element 1 added at the beginning



        for ele2 in range(predSeqLen):

            for ele1 in range(self.numLSTMLayers):

                itaLayerLPlus1InputGate = (self.WhhInputGate[ele1] @ hiddenStateTminus1LayerLplus1[ele1]) + (self.WxhInputGate[ele1] @ inputVector)
                itaLayerLPlus1ForgetGate = (self.WhhForgetGate[ele1] @ hiddenStateTminus1LayerLplus1[ele1]) + (self.WxhForgetGate[ele1] @ inputVector)
                itaLayerLPlus1OutputGate = (self.WhhOutputGate[ele1] @ hiddenStateTminus1LayerLplus1[ele1]) + (self.WxhOutputGate[ele1] @ inputVector)
                itaLayerLPlus1GateGate = (self.WhhGateGate[ele1] @ hiddenStateTminus1LayerLplus1[ele1]) + (self.WxhGateGate[ele1] @ inputVector)


                inputGate = self.mlffnn.activation_function(itaLayerLPlus1InputGate, 'sigmoid')
                forgetGate = self.mlffnn.activation_function(itaLayerLPlus1ForgetGate, 'sigmoid')
                outputGate = self.mlffnn.activation_function(itaLayerLPlus1OutputGate, 'sigmoid')
                gateGate = self.mlffnn.activation_function(itaLayerLPlus1GateGate, 'tanh')

                cellStateT = (forgetGate * cellStateTminus1[ele1]) + (inputGate * gateGate )
                hiddenStateTLayerLplus1 = outputGate * self.mlffnn.activation_function(cellStateT, 'tanh')


                cellStateTminus1[ele1] = cellStateT
                hiddenStateTminus1LayerLplus1[ele1] = hiddenStateTLayerLplus1
                inputVector = np.concatenate(([1],hiddenStateTLayerLplus1))

            inputToDNN = hiddenStateTLayerLplus1[:,None]
            self.mlffnn.forwardpass(inputToDNN, 'test')

            """Input vector/output after looping through all the LSTM layers and DNN is a probability distribution over
            the vocabulary
            """
            outputPMF = (self.mlffnn.predictedOutput).flatten()

            # Sample from this distribution
            values = np.arange(self.outputShape)
            chrIndex = np.random.choice(values, p=outputPMF)
            char = self.params['idx2char'][chrIndex]
            textString += char
            inputVector = np.zeros((self.inputShape+1,))
            inputVector[0] = 1 # 1st element is for the bias term
            inputVector[chrIndex+1] = 1 # To account for the element 1 added at the beginning

        print('Predicted text:\n',textString)





