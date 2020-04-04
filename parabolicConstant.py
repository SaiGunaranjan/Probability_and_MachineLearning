# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:09:17 2020

@author: saiguna


Problem Statement: If you were to pick up points randomly from a square grid of dimension 6 x 6 
and then find the distance of the point from the centre of the square, what is the mean distance 
of the points from the centre of the square grid. It turns out that the mean distance is 
actually the universal parabolic constant:  sqrt(2) + log(1+sqrt(2)) ~ 2.2955

For more details refer to the twitter thread by Tamas Gorbe:
    https://twitter.com/TamasGorbe/status/1246014582113492994



"""


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
numDataPoints = np.int32(1e7)
xCordinate = np.random.uniform(low=-3,high=+3,size=numDataPoints);
yCordinate = np.random.uniform(low=-3,high=+3,size=numDataPoints);
XYtuple = np.vstack((xCordinate,yCordinate)).T
distFromOrigin = np.linalg.norm(XYtuple,axis=1);
minDist = round(np.amin(distFromOrigin)*100)/100;
maxDist = round(np.amax(distFromOrigin)*100)/100;
numBins = np.int32(1e2)
counts,binEdges = np.histogram(distFromOrigin,bins=numBins);
binCentres = (binEdges[0:-1] + binEdges[1::])/2;
pmf = counts/np.sum(counts) # probability mass function
meanDistanceFromOrigin = np.sum(binCentres*pmf) # expectation = weighted sum of values
meanDistanceFromOrigin = round(np.amax(meanDistanceFromOrigin)*100)/100;
modeDistanceFromOrigin = round(binCentres[np.argmax(pmf)]*100)/100;


# plt.figure(1,figsize=(20,10));
# plt.title('Histogram of distance from origin');
# plt.hist(distFromOrigin,bins=numBins);
# # plt.axvline(minDist,color='k');
# # plt.axvline(maxDist,color='k');
# # plt.text(minDist,0,str(minDist))
# # plt.text(maxDist,0,str(maxDist))
# plt.xlabel('Distance from origin');
# plt.ylabel('Frequency of occurence');
# plt.grid(True)


plt.figure(1,figsize=(20,10));
plt.title('probability density of distance from origin');
plt.plot(binCentres,pmf);
plt.axvline(minDist,linewidth=2,color='k');
plt.axvline(maxDist,linewidth=2,color='k');
plt.axvline(meanDistanceFromOrigin,linewidth=4, color='g');
plt.axvline(modeDistanceFromOrigin,linewidth=4, color='b');
plt.text(minDist,np.amax(pmf)/2,'Min: ' + str(minDist),fontsize=12)
plt.text(maxDist,np.amax(pmf)/2,'Max: ' + str(maxDist),fontsize=12)
plt.text(meanDistanceFromOrigin,np.amax(pmf)/2,'Mean: ' + str(meanDistanceFromOrigin),fontsize=12);
plt.text(modeDistanceFromOrigin,np.amax(pmf)/2,'Mode: ' + str(modeDistanceFromOrigin),fontsize=12)
plt.xlabel('Distance from origin');
plt.ylabel('probability of occurence');
plt.grid(True)