# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:33:35 2020

@author: saiguna
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')


maxnumNeedles = np.int32(1e3)
needleLength = 5
lineSpacing = 2*needleLength
lineInd = 100
verticalSpacedLines = np.arange(-lineInd,lineInd,lineSpacing);
needleCentreInd_start = verticalSpacedLines[0] - needleLength/2;
needleCentreInd_end = verticalSpacedLines[-1] + needleLength/2;

step=10
numNeedles = np.arange(10,maxnumNeedles,step)


numMonteCarloRuns = 100
piApproxVec = np.zeros((len(numNeedles),),dtype=np.float32)
count = 0
for needNum in numNeedles:
    piApprox = 0
    for run in np.arange(numMonteCarloRuns):
        needleTheta = np.random.uniform(low=0,high=np.pi,size=needNum);
        needleCentre = np.random.uniform(low=needleCentreInd_start,high=needleCentreInd_end,size=needNum);
        numTouchingNeedles=0
        for ele in np.arange(needNum):
            temp1 = verticalSpacedLines >= (needleCentre[ele] - (needleLength/2)*np.sin(needleTheta[ele]));
            temp2 = verticalSpacedLines <= (needleCentre[ele] + (needleLength/2)*np.sin(needleTheta[ele]));
            temp = temp1*temp2;
            if np.sum(temp)==1: 
                numTouchingNeedles+=1
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
plt.plot(numNeedles,piApproxVec,'-o');
plt.xlabel('Total number of needles')
plt.ylabel('pi approximation');
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(numNeedles,probNeedleCross,'-o');
plt.xlabel('Total number of needles');
plt.ylabel('Probability of needle touching line')
plt.grid(True)
