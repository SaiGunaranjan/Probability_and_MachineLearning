# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 23:56:34 2024

@author: Sai Gunaranjan
"""

import numpy as np
import cupy as cp
from numba import cuda
from cnn_gpu_kernels.device_functions import convolution_2d_valid, convolution_2d_full




def cnn_convolve2d_parallel_ker_gpu(inputImage3d, kernelFunctions,convMode='valid'):
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
        convType = np.int32(2)
    elif convMode == 'full':
        outputHeight = inputHeight + kernelHeight - 1 # For "full" mode of convolution
        outputWidth = inputWidth + kernelWidth - 1 # For "full" mode of convolution
        convType = np.int32(1)


    convOutput = cp.zeros((numKernels,outputHeight,outputWidth,numDataPoints),dtype=cp.float32)
    thrdPerBlock = 4
    blkPerGrid = int(cp.ceil(numKernels/thrdPerBlock))
    inputImage3d = cp.asarray(inputImage3d,dtype=cp.float32)
    kernelFunctions = cp.asarray(kernelFunctions,dtype=cp.float32)
    scratchpad = cp.zeros((outputHeight, outputWidth,numKernels),dtype=cp.float32)
    pad_height = kernelHeight - 1
    pad_width = kernelWidth - 1
    padded_input = cp.zeros((inputHeight + 2 * pad_height, inputWidth + 2 * pad_width, numKernels), dtype=cp.float32)
    convolve2d_parallel_ker_gpu[blkPerGrid,thrdPerBlock](inputImage3d,kernelFunctions,convOutput,numDataPoints,numKernels,numChannels, scratchpad, padded_input, convType)
    # cp.cuda.Device().synchronize()
    # convOutput = cp.array(convOutput)
    convOutput = cp.asnumpy(convOutput)

    return convOutput



def cnn_backward_convolve2d_parallel_chan_gpu(inputImage3d, kernelFunctions,convMode='valid'):
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
        convType = np.int32(2)
    elif convMode == 'full':
        outputHeight = inputHeight + kernelHeight - 1 # For "full" mode of convolution
        outputWidth = inputWidth + kernelWidth - 1 # For "full" mode of convolution
        convType = np.int32(1)

    convOutput = cp.zeros((numChannels,outputHeight,outputWidth,numDataPoints),dtype=cp.float32)
    thrdPerBlock = 4
    blkPerGrid = int(cp.ceil(numChannels/thrdPerBlock))
    inputImage3d = cp.asarray(inputImage3d,dtype=cp.float32)
    kernelFunctions = cp.asarray(kernelFunctions,dtype=cp.float32)
    scratchpad = cp.zeros((outputHeight, outputWidth,numChannels),dtype=cp.float32)
    pad_height = kernelHeight - 1
    pad_width = kernelWidth - 1
    padded_input = cp.zeros((inputHeight + 2 * pad_height, inputWidth + 2 * pad_width, numChannels), dtype=cp.float32)
    convolve2d_backward_parallel_chan_gpu[blkPerGrid,thrdPerBlock](inputImage3d,kernelFunctions,convOutput,numDataPoints,numKernels,numChannels, scratchpad, padded_input, convType)
    # cp.cuda.Device().synchronize()
    # convOutput = cp.array(convOutput)
    convOutput = cp.asnumpy(convOutput)

    return convOutput



def cnn_gradient_convolve2d_parallel_ker_gpu(outputLayerL, errorConvLayerLplu1,convMode='valid'):
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
        convType = np.int32(2)
    elif convMode == 'full':
        outputHeight = inputHeight + kernelHeight - 1 # For "full" mode of convolution
        outputWidth = inputWidth + kernelWidth - 1 # For "full" mode of convolution
        convType = np.int32(1)


    convOutput = cp.zeros((numKernels,outputHeight,outputWidth,numChannels, numDataPoints),dtype=cp.float32)
    thrdPerBlock = 4
    blkPerGrid = int(cp.ceil(numKernels/thrdPerBlock))
    outputLayerL = cp.asarray(outputLayerL,dtype=cp.float32)
    errorConvLayerLplu1 = cp.asarray(errorConvLayerLplu1,dtype=cp.float32)
    pad_height = kernelHeight - 1
    pad_width = kernelWidth - 1
    padded_input = cp.zeros((inputHeight + 2 * pad_height, inputWidth + 2 * pad_width, numKernels), dtype=cp.float32)
    convolve2d_gradient_parallel_ker_gpu[blkPerGrid,thrdPerBlock](outputLayerL,errorConvLayerLplu1,convOutput,numDataPoints,numKernels,numChannels, padded_input, convType)
    # cp.cuda.Device().synchronize()
    # convOutput = cp.array(convOutput)
    convOutput = cp.asnumpy(convOutput)

    return convOutput




""" Parallelize across kernels taking more time than parallelize across datapoints!!"""


@cuda.jit('void(float32[:,:,:,:],float32[:,:,:,:],float32[:,:,:,:], int32, int32, int32, float32[:,:,:], float32[:,:,:], int32)')
def convolve2d_parallel_ker_gpu(inputImage3d,kernelFunctions,convOutput,numDataPoints,numKernels,numChannels,scratchpad,padded_input,convType):


    thrdIDx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if ((thrdIDx < 0) or (thrdIDx >= numKernels)):
        return;

    for ele3 in range(numDataPoints):
        for ele2 in range(numChannels):

            if (convType == 1):
                convolution_2d_full(inputImage3d[ele2,:,:,ele3], kernelFunctions[thrdIDx,:,:,ele2], scratchpad[:,:,thrdIDx], padded_input[:,:,thrdIDx])
            elif (convType == 2):
                convolution_2d_valid(inputImage3d[ele2,:,:,ele3], kernelFunctions[thrdIDx,:,:,ele2], scratchpad[:,:,thrdIDx])

            for i in range(scratchpad.shape[0]):
                for j in range(scratchpad.shape[1]):
                    convOutput[thrdIDx,i,j,ele3] = convOutput[thrdIDx,i,j,ele3] + scratchpad[i,j,thrdIDx]




""" Parallelize across channels"""
@cuda.jit('void(float32[:,:,:,:],float32[:,:,:,:],float32[:,:,:,:], int32, int32, int32, float32[:,:,:], float32[:,:,:], int32)')
def convolve2d_backward_parallel_chan_gpu(inputImage3d,kernelFunctions,convOutput,numDataPoints,numKernels,numChannels,scratchpad,padded_input,convType):


    thrdIDx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if ((thrdIDx < 0) or (thrdIDx >= numChannels)):
        return;

    for ele3 in range(numDataPoints):
        for ele2 in range(numKernels):
            if (convType == 1):
                convolution_2d_full(inputImage3d[ele2,:,:,ele3], kernelFunctions[ele2,:,:,thrdIDx], scratchpad[:,:,thrdIDx], padded_input[:,:,thrdIDx])
            elif (convType == 2):
                convolution_2d_valid(inputImage3d[ele2,:,:,ele3], kernelFunctions[ele2,:,:,thrdIDx], scratchpad[:,:,thrdIDx])

            for i in range(scratchpad.shape[0]):
                for j in range(scratchpad.shape[1]):
                    convOutput[thrdIDx,i,j,ele3] = convOutput[thrdIDx,i,j,ele3] + scratchpad[i,j,thrdIDx]



""" Parallelize across kernels"""
@cuda.jit('void(float32[:,:,:,:],float32[:,:,:,:],float32[:,:,:,:,:], int32, int32, int32, float32[:,:,:], int32)')
def convolve2d_gradient_parallel_ker_gpu(outputLayerL,errorConvLayerLplu1,convOutput,numDataPoints,numKernels,numChannels,padded_input,convType):


    thrdIDx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if ((thrdIDx < 0) or (thrdIDx >= numKernels)):
        return;

    for ele3 in range(numDataPoints):
        for ele2 in range(numChannels):

            if (convType == 1):
                convolution_2d_full(outputLayerL[ele2,:,:,ele3], errorConvLayerLplu1[thrdIDx,:,:,ele3], convOutput[thrdIDx,:,:,ele2,ele3], padded_input[:,:,thrdIDx])
            elif (convType == 2):
                convolution_2d_valid(outputLayerL[ele2,:,:,ele3], errorConvLayerLplu1[thrdIDx,:,:,ele3], convOutput[thrdIDx,:,:,ele2,ele3])
