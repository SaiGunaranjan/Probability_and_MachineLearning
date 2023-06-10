# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 23:45:29 2023

@author: Sai Gunaranjan
"""


""" This code contains the 1d implementation of the popular Expectation Maximization(EM) algorithm.
I have implemented this algorithm based on the videos of Arun Rajkumar of IIT Madras. The algorithm is
beautifully explained in the video lectures. I have also referred the EM video lecture by Alexander Ihler for the code implementation part.
The derivation of this algo is available in Arun's videos and also in the CS229 course notes of Andrew NG.
In the course notes, he explains how the convex combination used in the Jensen's inequality can be interpreted as an Expectation.
For a concave function f(x), f(E[x]) >= E[f(x)]. This is a very crucial interpretation and this is how the Log likelihood of a mixture of
gaussians is converted to a simple form which can be analytically computed"""

import numpy as np
import matplotlib.pyplot as plt


plt.close('all')
""" Data generation using Gaussian Mixture Model"""
numClusters = 2
priors = 0.5*np.ones((numClusters))
meanArray = np.array([7,15])
varianceArray = np.array([1,1])
priorsArray = np.array([0.8,0.2])
numDataPoints = 20000
numDataPointsEachCluster = (priorsArray * numDataPoints).astype('int32') #np.array([10000,10000])
numDataPoints = np.sum(numDataPointsEachCluster)

dataPointsArray = np.empty([0])
plt.figure(1,figsize=(20,10))
for ele in range(numClusters):
    dataPoints = np.random.normal(meanArray[ele], varianceArray[ele], numDataPointsEachCluster[ele])
    count,bins,_ = plt.hist(dataPoints,bins=100,density=True)
    # plt.plot(bins, priorsArray[ele] * 1/(np.sqrt(2 * np.pi * varianceArray[ele]**2)) * np.exp( - (bins - meanArray[ele])**2 / (2 * varianceArray[ele]**2) ),linewidth=2, color='black')
    plt.plot(bins, 1/(np.sqrt(2 * np.pi * varianceArray[ele]**2)) * np.exp( - (bins - meanArray[ele])**2 / (2 * varianceArray[ele]**2) ),linewidth=2, color='black')
    plt.grid(True)
    dataPointsArray = np.append(dataPointsArray,dataPoints)


plt.figure(2,figsize=(20,10))
plt.title('Data distribution')
count,bins,_ = plt.hist(dataPointsArray,bins=200,density=True)
plt.grid(True)


dataPointsArray = np.sort(dataPointsArray) # Sorting the datapoints for subsequenct plotting convenience

"""EM algorithm """

""" Step 1: Initialization of parameters"""
meanArrayEM = np.array([4,17]) # np.array([-5,10])
varianceArrayEM = np.array([2,2])
priorProbArrayEM = np.array([0.5,0.5])

print('Iteration: {}'.format(0))
print('Mean', meanArrayEM)
print('Variance', varianceArrayEM)
print('\n')

numIterations = 15
logLikelihoodArray = np.zeros((numIterations))
plt.figure(3,figsize=(20,10))
for iterCount in range(numIterations):
    """ Step 2: E step. Compute the posteriors(labda_ij) of the latent random variables (P(z/x))"""
    """ For this, we need to compute P(x/z)"""

    Pxcondz =   1/(np.sqrt(2 * np.pi * varianceArrayEM[None,:]**2)) * np.exp( - (dataPointsArray[:,None] - meanArrayEM[None,:])**2 / (2 * varianceArrayEM[None,:]**2))
    Px = np.sum(Pxcondz,axis=1)
    Pzcondx =  priorProbArrayEM[None,:] * (Pxcondz / Px[:,None]) # lamda_ij (index i for the data point and index j for the cluster)
    logLikelihood = np.sum(np.log10(Px))
    logLikelihoodArray[iterCount] = logLikelihood

    """ Step 3: M step. Re-estimate the parameters using the posteriors"""
    meanArrayEM = np.sum(dataPointsArray[:,None] * Pzcondx, axis=0)/ np.sum(Pzcondx,axis=0)
    varianceArrayEM = np.sum(((dataPointsArray[:,None] - meanArrayEM[None,:])**2) * Pzcondx,axis=0)/ np.sum(Pzcondx,axis=0)
    priorProbArrayEM = np.sum(Pzcondx,axis=0)/numDataPoints

    print('Iteration: {}'.format(iterCount+1))
    print('Mean', meanArrayEM)
    print('Variance', varianceArrayEM)
    print('Priors', priorProbArrayEM)
    print('\n')

    plt.clf()
    plt.subplot(1,2,1)
    plt.title('P(x/z)')
    # plt.plot(bins, 1/(np.sqrt(2 * np.pi * varianceArrayEM[None,:]**2)) * np.exp( - (bins[:,None] - meanArrayEM[None,:])**2 / (2 * varianceArrayEM[None,:]**2) ),linewidth=2, color='black')
    plt.plot(dataPointsArray,Pxcondz/np.sum(Pxcondz,axis=0)[None,:])
    plt.ylim([0,2e-4])
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.title('P(z/x) in log scale')
    plt.plot(dataPointsArray,np.log10(Pzcondx))
    plt.ylim([-50,0])
    plt.grid(True)

    plt.pause(1)


plt.figure(4,figsize=(20,10))
plt.title('Log likelihood')
plt.plot(logLikelihoodArray,'-o')
plt.xlabel('Iteration number')
plt.grid(True)
