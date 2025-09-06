# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:17:59 2025

@author: Sai Gunaranjan
"""

"""
Word-level language model using the custom RNN.
This mirrors the character-level script but switches preprocessing to tokens='word'.
"""

from rnn_class import RecurrentNeuralNetwork
from textfile_preprocessing import load_text_file, tokenize_text, create_vocab
import time as time


textfilepath = "harry_potter_small.txt"
text = load_text_file(textfilepath)
char_or_word = 'char' # char_or_word='word' or 'char'
tokens = tokenize_text(text, level=char_or_word, lowercase=True, keep_punct=True)
token2idx, idx2token = create_vocab(tokens)
vocab_size = len(idx2token)

inputShape = vocab_size
# bump hidden size for word-level.
if char_or_word == 'char':
    hiddenStateVecLengthEachRNNLayer = [100]
else:
    hiddenStateVecLengthEachRNNLayer = [500] # [2*vocab_size]
outputShape = vocab_size
numTimeSteps = 25 #300


rnn = RecurrentNeuralNetwork(inputShape, hiddenStateVecLengthEachRNNLayer, outputShape, numTimeSteps)
rnn.set_model_params(batchsize = 1, epochs=1000, stepsize=1e-3) # epochs=1000, stepsize=1e-1
# Key change: level='word'
rnn.preprocess_textfile(textfilepath, level=char_or_word, lowercase=True, keep_punct=True)

tstart = time.time()
rnn.train() # Data is already available in the rnn class and split=0.8 is already defined in the text_prepocessing file

predSeqLen = 2000
rnn.predict(predSeqLen) # Generates a character sequence of length = predSeqLen

tend = time.time()

timeTrainTest = (tend - tstart)/(60*60)
print('\n\nTotal time taken for training and testing  = {0:.2f} hours'.format(timeTrainTest))
