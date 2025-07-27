# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:17:59 2025

@author: Sai Gunaranjan
"""


from recurrent_neural_network import RecurrentNeuralNetwork
from textfile_preprocessing import load_text_file, create_vocab
import time as time


textfilepath = "harry_potter_small.txt"
text = load_text_file(textfilepath)
char2idx, idx2char = create_vocab(text)
vocab_size = len(idx2char)

inputShape = vocab_size
numRNNLayers = 1
outputShape = vocab_size
numTimeSteps = 25


rnn = RecurrentNeuralNetwork(inputShape, numRNNLayers, outputShape, numTimeSteps)
rnn.set_model_params(batchsize = 1, epochs=5000, stepsize=1e-3) # epochs=1000, stepsize=1e-1
rnn.preprocess_textfile(textfilepath)

tstart = time.time()
rnn.train() # Data is already available in the rnn class and split=0.8 is already defined in the text_prepocessing file

predSeqLen = 2000
rnn.predict(predSeqLen) # Generates a character sequence of length = predSeqLen

tend = time.time()

timeTrainTest = (tend - tstart)/(60*60)
print('\n\nTotal time taken for training and testing  = {0:.2f} hours'.format(timeTrainTest))
