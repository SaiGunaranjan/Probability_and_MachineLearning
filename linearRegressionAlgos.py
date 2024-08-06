# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:51:29 2022

@author: Sai Gunaranjan
"""

"""

Implemented the batch and online/stochastic versions of the gradient descent algorithm to solve
the linear least squares problem and compared against the pseudo inverse solution.
I have given a flag to choose between the batch/online version of the gradient descent algorithm to solve
the linear least squares problem.



1. Make epochs on x axis and not each data point. Get the MSE for each epoch.
2. Plot the estimated line for each iteration to check how it is changing for each iteration/epoch
3. Solve other variants of the least squares like ridge regression, regularized least squares, and also the LMS algorithm
"""


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

plt.close('all')

slopeDeg = 45#0#45
slopeRad = slopeDeg*np.pi/180
slope = np.tan(slopeRad)
constant = 80
lineParameters = np.array([constant, slope])

numDataPoints = 100
xVals = np.random.rand(numDataPoints)#np.random.randint(low=-20,high=100,size=100)#np.arange(numDataPoints)
dataPoints = np.hstack((np.ones(numDataPoints)[:,None], xVals[:,None]))
dataValues = dataPoints @ lineParameters
sigma = 2#10
noise = sigma*np.random.randn(numDataPoints)
noisyDataValues = dataValues + noise



flagBatchGradDes = False#True # False implies stochastic grad descent/online gradient descent

if flagBatchGradDes:
    """ Batch Gradient descent init parameters"""
    lineParametersInit = np.array([75,-5]) #np.array([8,0.5])
    estParamsBatchGradDes = lineParametersInit
    alphaBatGradDes = 1e-3#1e-3
    convergenceThreshold = 1e-8#0.001
    numEpochs = 500
else:
    """ Online/Stochastic Gradient descent init parameters"""
    lineParametersInit = np.array([75,-5]) #np.array([8,0.5])
    estParamsBatchGradDes = lineParametersInit
    alphaBatGradDes = 1e-1#1e-3
    convergenceThreshold = 1e-8#0.001
    numEpochs = 10

costFunctionVector = np.empty([0])
residualEnergyVector = np.empty([0])
estParams = np.zeros((0,2),dtype=np.float32)
iterCount = 0

arr = np.arange(numDataPoints)

while iterCount<numEpochs:

    if flagBatchGradDes:
        costFunction = np.linalg.norm(dataPoints@estParamsBatchGradDes - noisyDataValues)**2
        costFunctionVector = np.append(costFunctionVector,costFunction)

        gradient = dataPoints.T @ (dataPoints@estParamsBatchGradDes - noisyDataValues)
        estParamsBatchGradDesUpdated = estParamsBatchGradDes - alphaBatGradDes*gradient
        residualEnergy = np.linalg.norm(estParamsBatchGradDesUpdated-estParamsBatchGradDes)**2
        residualEnergyVector = np.append(residualEnergyVector,residualEnergy)
        estParamsBatchGradDes = estParamsBatchGradDesUpdated

        # print('Cost function val = ', np.round(costFunction,2))
        # print('Residual energy val = ', np.round(residualEnergy,2))
        print('c = {0:.2f}, m = {1:.2f} \n'.format(estParamsBatchGradDes[0],estParamsBatchGradDes[1]))

        estParams = np.vstack((estParams,estParamsBatchGradDes[None,:]))



        if residualEnergy < convergenceThreshold:
            break

        plt.figure(1,figsize=(20,10),dpi=150)
        plt.clf()

        plt.subplot(1,3,1)
        plt.title('Batch Gradient descent')
        plt.xlim([constant-10,constant+10])
        plt.ylim([slope-10,slope+10])
        plt.xlabel('constant')
        plt.ylabel('slope')
        plt.scatter(constant, slope,label='GT')
        plt.scatter(estParams[:,0], estParams[:,1],alpha=0.5)
        plt.grid(True)
        plt.legend()

        plt.subplot(1,3,2)
        plt.title('Cost function energy (dB)')
        plt.plot(10*np.log10(costFunctionVector))
        plt.grid(True)
        plt.xlim([0,numEpochs])
        plt.xlabel('Iteration #')

        plt.subplot(1,3,3)
        plt.title('Residual Energy (dB)')
        plt.plot(10*np.log10(residualEnergyVector))
        plt.grid(True)
        plt.xlim([0,numEpochs])
        plt.xlabel('Iteration #')

        plt.pause(0.01)

        iterCount += 1

    else:
        np.random.shuffle(arr)
        for ele in arr:
            costFunction = np.linalg.norm(dataPoints[ele,:]@estParamsBatchGradDes - noisyDataValues[ele])**2
            costFunctionVector = np.append(costFunctionVector,costFunction)

            gradient = dataPoints[ele,:] * (dataPoints[ele,:]@estParamsBatchGradDes - noisyDataValues[ele])
            estParamsBatchGradDesUpdated = estParamsBatchGradDes - alphaBatGradDes*gradient
            residualEnergy = np.linalg.norm(estParamsBatchGradDesUpdated-estParamsBatchGradDes)**2
            residualEnergyVector = np.append(residualEnergyVector,residualEnergy)
            estParamsBatchGradDes = estParamsBatchGradDesUpdated

            # print('Cost function val = ', np.round(costFunction,2))
            # print('Residual energy val = ', np.round(residualEnergy,2))
            print('c = {0:.2f}, m = {1:.2f} \n'.format(estParamsBatchGradDes[0],estParamsBatchGradDes[1]))

            estParams = np.vstack((estParams,estParamsBatchGradDes[None,:]))

            if residualEnergy < convergenceThreshold:
                break


            plt.figure(1,figsize=(20,10),dpi=150)
            plt.clf()

            plt.subplot(1,3,1)
            plt.title('Stochastic Gradient descent')
            plt.xlim([constant-10,constant+10])
            plt.ylim([slope-10,slope+10])
            plt.xlabel('constant')
            plt.ylabel('slope')
            plt.scatter(constant, slope,label='GT')
            plt.scatter(estParams[:,0], estParams[:,1],alpha=0.5)
            plt.grid(True)
            plt.legend()

            plt.subplot(1,3,2)
            plt.title('Cost function energy (dB)')
            plt.plot(10*np.log10(costFunctionVector))
            plt.grid(True)
            plt.xlim([0,numDataPoints*numEpochs])
            plt.xlabel('Iteration #')

            plt.subplot(1,3,3)
            plt.title('Residual Energy (dB)')
            plt.plot(10*np.log10(residualEnergyVector))
            plt.grid(True)
            plt.xlim([0,numDataPoints*numEpochs])
            plt.xlabel('Iteration #')

            plt.pause(0.01)


        iterCount += 1
        print('\nEpoch {0} / {1} completed\n'.format(iterCount, numEpochs))


""" Single shot solution using Least squares"""
estLineParametersLstSq,_,_,_ = np.linalg.lstsq(dataPoints, noisyDataValues,rcond=None)
dataValuesFromLstSq = dataPoints @ estLineParametersLstSq


dataValuesFromGradDes = dataPoints @ estParamsBatchGradDes

print('Actual values: ', lineParameters)
print('With Least Squares: ', estLineParametersLstSq)
print('With Gradient Descent: ', estParamsBatchGradDes)


plt.figure(2,figsize=(20,10),dpi=200)
plt.plot(xVals, dataValues,color='k',lw=2,label='GT')
plt.scatter(xVals,noisyDataValues,label='noisy values')
plt.plot(xVals, dataValuesFromLstSq,color='orange',lw=4,alpha=0.6,label='Least Sq soln')
plt.plot(xVals, dataValuesFromGradDes,color='green',lw=4,alpha=0.6,label='Grad desc soln')
plt.grid(True)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(min(xVals)-1,max(xVals)+1)
plt.ylim([min(dataValues)-1,max(dataValues)+1])