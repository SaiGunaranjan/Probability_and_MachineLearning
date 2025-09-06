# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:17:59 2025

@author: Sai Gunaranjan
"""

"""
Word-level language model using the custom LSTM.
This mirrors the character-level script but switches preprocessing to tokens='word'.
"""

from lstm_class import LSTM
from textfile_preprocessing import load_text_file, tokenize_text, create_vocab
import time as time

# Build vocab at word level to get input/output shapes
textfilepath = "harry_potter_small.txt"
text = load_text_file(textfilepath)
char_or_word = 'char' # char_or_word='word' or 'char'
tokens = tokenize_text(text, level=char_or_word, lowercase=True, keep_punct=True)
token2idx, idx2token = create_vocab(tokens)
vocab_size = len(idx2token)

inputShape = vocab_size
hiddenStateVecLengthEachLSTMLayer = [100] # [2*vocab_size]  # bump hidden size for word-level. 100 for char level and 200 for word level as an example.
denseLayer = []
numOutputNodes = vocab_size
outputLayer = [(numOutputNodes,'softmax',0)]
numTimeSteps = 25  # number of tokens per unroll


lstm = LSTM(inputShape, hiddenStateVecLengthEachLSTMLayer, denseLayer, outputLayer, numTimeSteps)
lstm.mlffnn.set_model_params(modeGradDescent = 'mini_batch',batchsize = 1, costfn = 'categorical_cross_entropy',epochs = 2000, stepsize = 1e-2)
# Key change: level='word'
lstm.preprocess_textfile(textfilepath, level=char_or_word, lowercase=True, keep_punct=True)


tstart = time.time()
lstm.train() # Data is already available in the LSTM class and split=0.8 is already defined in the text_prepocessing file

predSeqLen = 2000
lstm.predict(predSeqLen) # Generates a character sequence of length = predSeqLen

tend = time.time()

timeTrainTest = (tend - tstart)/(60*60)
print('\n\nTotal time taken for training and testing  = {0:.2f} hours'.format(timeTrainTest))
