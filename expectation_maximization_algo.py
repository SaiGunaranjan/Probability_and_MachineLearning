# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 23:45:29 2023

@author: Sai Gunaranjan
"""

import numpy as np
import matplotlib.pyplot as plt


plt.close('all')
""" Data generation using Gaussian Mixture Model"""
numClusters = 2
priors = 0.5*np.ones((numClusters))
meanArray = np.array([-10,15])
varianceArray = np.array([1,1])
numDataPointsEachCluster = np.array([10000,10000])


dataPointsArray = np.empty([0])
plt.figure(1,figsize=(20,10))
for ele in range(numClusters):
    dataPoints = np.random.normal(meanArray[ele], varianceArray[ele], numDataPointsEachCluster[ele])
    count,bins,_ = plt.hist(dataPoints,bins=100,density=True)
    plt.plot(bins, 1/(np.sqrt(2 * np.pi * varianceArray[ele]**2)) * np.exp( - (bins - meanArray[ele])**2 / (2 * varianceArray[ele]**2) ),linewidth=2, color='black')
    plt.grid(True)
    dataPointsArray = np.append(dataPointsArray,dataPoints)


plt.figure(2,figsize=(20,10))
plt.title('Data distribution')
count,bins,_ = plt.hist(dataPointsArray,bins=200,density=True)
plt.grid(True)
# plt.close('all')

dataPointsArray = np.sort(dataPointsArray) # Sorting the datapoints for subsequenct plotting convenience
"""EM algorithm """

""" Step 1: Initializtion of parameters"""
meanArrayEM = np.array([-20,20]) # np.array([-5,10])
varianceArrayEM = np.array([2,2])
plt.figure(3,figsize=(20,10))
for iterCount in range(15):
    """ Step 2: Compute the posteriors of the latent random variables (P(z/x))"""
    """ For this, we need to compute P(x/z)"""

    Pxcondz =  1/(np.sqrt(2 * np.pi * varianceArrayEM[None,:]**2)) * np.exp( - (dataPointsArray[:,None] - meanArrayEM[None,:])**2 / (2 * varianceArrayEM[None,:]**2))
    Pzcondx = Pxcondz/np.sum(Pxcondz,axis=1)[:,None]

    """ Step 3: Re-estimate the parameters using the posteriors"""
    meanArrayEM = np.sum(dataPointsArray[:,None] * Pzcondx, axis=0)/ np.sum(Pzcondx,axis=0)
    varianceArrayEM = np.sum(((dataPointsArray[:,None] - meanArrayEM[None,:])**2) * Pzcondx,axis=0)/ np.sum(Pzcondx,axis=0)
    print('Iteration: {}'.format(iterCount+1))
    print('Mean', meanArrayEM)
    print('Variance', varianceArrayEM)
    print('\n')
    plt.clf()
    plt.subplot(1,2,1)
    plt.title('P(x/z)')
    # plt.plot(bins, 1/(np.sqrt(2 * np.pi * varianceArrayEM[None,:]**2)) * np.exp( - (bins[:,None] - meanArrayEM[None,:])**2 / (2 * varianceArrayEM[None,:]**2) ),linewidth=2, color='black')
    plt.plot(dataPointsArray,Pxcondz/np.sum(Pxcondz))
    # plt.ylim([-0.25,1])
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.title('P(z/x)')
    plt.plot(dataPointsArray,Pzcondx)
    plt.ylim([-0.25,1.5])
    plt.grid(True)

    plt.pause(1)


