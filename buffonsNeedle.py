# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:33:35 2020

@author: saiguna
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

maxnumNeedles = np.int32(1e6)
needleLength = 5
lineSpacing = 2*needleLength
lineInd = 100
verticalSpacedLines = np.arange(-lineInd,lineInd,lineSpacing);
needleCentreInd_start = verticalSpacedLines[0];
needleCentreInd_end = verticalSpacedLines[-1];

step = np.int32(1e4)
numNeedles = np.arange(10,maxnumNeedles,step)

numMonteCarloRuns = 10
piApproxVec = np.zeros((len(numNeedles),),dtype=np.float32)
count = 0
for needNum in numNeedles:
    piApprox = 0
    for run in np.arange(numMonteCarloRuns):
        needleTheta = np.random.uniform(low=0,high=np.pi,size=needNum);
        needleCentre = np.random.uniform(low=needleCentreInd_start,high=needleCentreInd_end,size=needNum);
    
        temp1 = verticalSpacedLines[None,:] >= (needleCentre - (needleLength/2)*np.sin(needleTheta))[:,None];
        temp2 = verticalSpacedLines[None,:] <= (needleCentre + (needleLength/2)*np.sin(needleTheta))[:,None];
        temp = temp1 * temp2;
        numTouchingNeedles = np.sum(temp)
        piApprox += (2*needleLength/lineSpacing)*(needNum/numTouchingNeedles)
        
    piApproxVec[count] = piApprox/numMonteCarloRuns
    count+=1

probNeedleCross = (2*needleLength/lineSpacing)/piApproxVec


# plt.figure(1,figsize=(20,10));
# plt.hlines(verticalSpacedLines,xmin=0,xmax=numNeedles);
# plt.plot(needleCentre,'o');
# plt.grid(True)

plt.figure(2,figsize=(20,10));
plt.subplot(1,2,1)
plt.title('Estimated pi value');
plt.plot(numNeedles,piApproxVec,'-o');
plt.axhline(np.pi,color='k',linewidth=2)
plt.xlabel('Total number of needles')

plt.grid(True)
plt.subplot(1,2,2)
plt.title('Probability of needle touching line')
plt.plot(numNeedles,probNeedleCross,'-o');
plt.axhline(1/np.pi,color='k')
plt.xlabel('Total number of needles');
plt.grid(True)
