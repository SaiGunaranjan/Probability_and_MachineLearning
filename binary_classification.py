# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:40:35 2023

@author: Sai Gunaranjan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from ml_functions_lib import perceptron_train, perceptron_test, perceptron_accuracy,\
    logistic_regression_train, logistic_regression_test, logistic_regression_accuracy,\
        svm_train, svm_test, svm_accuracy
from sklearn import svm


np.random.seed(10)
plt.close('all')

""" Create the dataset for binary class"""
# Data, labels = datasets.make_classification(n_samples=1000,n_features=2,n_classes=2,n_clusters_per_class=1,n_redundant=0,\
#                                             class_sep=1) # random_state = 2, class_sep =1 causing issues for logistic regression

Data, labels = datasets.make_blobs(n_samples=1000,n_features=2,centers=2) # random_state = 2, class_sep =1 causing issues for logistic regression

numDataPoints = Data.shape[0]
numFeatures = Data.shape[1]
numTrainingData = int(np.round(0.7 * numDataPoints))

trainingData = Data[0:numTrainingData,:]
trainingLabels = labels[0:numTrainingData]

trainingDataClass0 = trainingData[trainingLabels==0,:]
trainingDataClass1 = trainingData[trainingLabels==1,:]

testingData = Data[numTrainingData::,:]
testingLabels = labels[numTrainingData::]

testingDataClass0 = testingData[testingLabels==0,:]
testingDataClass1 = testingData[testingLabels==1,:]


""" Perceptron"""
""" Convert the labels from 0/1 to -1/1 to handle the update equation in the perceptron algorithm"""
trainingLabels_perceptron = trainingLabels.copy()
trainingLabels_perceptron[trainingLabels_perceptron==0] -= 1

testingLabels_perceptron = testingLabels.copy()
testingLabels_perceptron[testingLabels_perceptron==0] -= 1

wVec_perceptron = perceptron_train(trainingData,trainingLabels_perceptron)
estLabels_perceptron = perceptron_test(testingData,wVec_perceptron)
perceptron_accuracy(testingLabels_perceptron, estLabels_perceptron)


""" Logistic regression"""
wVec_logReg, logLikelihood = logistic_regression_train(trainingData,trainingLabels)
estLabels_logReg = logistic_regression_test(testingData,wVec_logReg)
logistic_regression_accuracy(testingLabels, estLabels_logReg)

""" Support Vector machine"""
trainingLabels_svm = trainingLabels_perceptron.copy()
testingLabels_svm = testingLabels_perceptron.copy()

""" Remove bias/mean in the data"""
# trainingData -= np.mean(trainingData,axis=0)[None,:]
# testingData -= np.mean(testingData,axis=0)[None,:]

wVec_svm, costFunctionDualProblem = svm_train(trainingData,trainingLabels_svm)
estLabels_svm = svm_test(testingData,wVec_svm)
svm_accuracy(testingLabels_svm, estLabels_svm)


""" SVM library"""
# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(trainingData, trainingLabels_svm)

# plt.figure(6,figsize=(20,10),dpi=200)
# plt.scatter(trainingData[:, 0], trainingData[:, 1], c=trainingLabels_svm, s=30, cmap=plt.cm.Paired)

# # plot the decision function
# ax = plt.gca()
# # DecisionBoundaryDisplay.from_estimator(
# #     clf,
# #     X,
# #     plot_method="contour",
# #     colors="k",
# #     levels=[-1, 0, 1],
# #     alpha=0.5,
# #     linestyles=["--", "-", "--"],
# #     ax=ax,
# # )
# # plot support vectors
# ax.scatter(
#     clf.support_vectors_[:, 0],
#     clf.support_vectors_[:, 1],
#     s=100,
#     linewidth=1,
#     facecolors="none",
#     edgecolors="k",
# )
# plt.show()



""" Plotting the separating hyper plane"""
separatingLineXcordinate = np.linspace(np.amin(Data[:,0])-2,np.amax(Data[:,0])+2,100)
separatingLineYcordinate_perceptron = (-wVec_perceptron[-1] - wVec_perceptron[0]*separatingLineXcordinate)/wVec_perceptron[1]
separatingLineYcordinate_logReg = (-wVec_logReg[-1] - wVec_logReg[0]*separatingLineXcordinate)/wVec_logReg[1]
separatingLineYcordinate_svm = (-wVec_svm[-1] - wVec_svm[0]*separatingLineXcordinate)/wVec_svm[1]
separatingLineYcordinate_svm_supportVec1 = (1-wVec_svm[-1] - wVec_svm[0]*separatingLineXcordinate)/wVec_svm[1]
separatingLineYcordinate_svm_supportVec2 = (-1-wVec_svm[-1] - wVec_svm[0]*separatingLineXcordinate)/wVec_svm[1]

separatingLineYcordinate_svm = (-wVec_svm[-1] - wVec_svm[0]*separatingLineXcordinate)/wVec_svm[1]
separatingLineYcordinate_svmlib = (-clf.intercept_ - clf.coef_[0][0]*separatingLineXcordinate)/clf.coef_[0][1]
separatingLineYcordinate_svmlib_supportVec1 = (1-clf.intercept_ - clf.coef_[0][0]*separatingLineXcordinate)/clf.coef_[0][1]
separatingLineYcordinate_svmlib_supportVec2 = (-1-clf.intercept_ - clf.coef_[0][0]*separatingLineXcordinate)/clf.coef_[0][1]

print('Gamma margin from my SVM = {0:.2f}, from sklearn = {1:.2f}'.format(2 / np.linalg.norm(wVec_svm[0:2])**2, 2 / np.linalg.norm(clf.coef_[0,:])**2))

if 0:
    plt.figure(1,figsize=(20,10),dpi=200)
    plt.suptitle('Perceptron')
    plt.subplot(1,2,1)
    plt.title('Training data')
    plt.scatter(trainingDataClass0[:,0],trainingDataClass0[:,1],color='red')
    plt.scatter(trainingDataClass1[:,0],trainingDataClass1[:,1],color='green')
    plt.plot(separatingLineXcordinate,separatingLineYcordinate_perceptron)
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
    plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])
    # plt.axis('scaled')

    plt.subplot(1,2,2)
    plt.title('Testing data')
    plt.scatter(testingDataClass0[:,0],testingDataClass0[:,1],color='red')
    plt.scatter(testingDataClass1[:,0],testingDataClass1[:,1],color='green')
    plt.plot(separatingLineXcordinate,separatingLineYcordinate_perceptron)
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
    plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])


    plt.figure(2,figsize=(20,10),dpi=200)
    plt.suptitle('Logistic Regression')
    plt.subplot(1,2,1)
    plt.title('Training data')
    plt.scatter(trainingDataClass0[:,0],trainingDataClass0[:,1],color='red')
    plt.scatter(trainingDataClass1[:,0],trainingDataClass1[:,1],color='green')
    plt.plot(separatingLineXcordinate,separatingLineYcordinate_logReg)
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
    plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])
    # plt.axis('scaled')

    plt.subplot(1,2,2)
    plt.title('Testing data')
    plt.scatter(testingDataClass0[:,0],testingDataClass0[:,1],color='red')
    plt.scatter(testingDataClass1[:,0],testingDataClass1[:,1],color='green')
    plt.plot(separatingLineXcordinate,separatingLineYcordinate_logReg)
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
    plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])


    plt.figure(3,figsize=(20,10),dpi=200)
    plt.title('Logistic regression: Log likelihood vs iterations')
    plt.plot(logLikelihood[1::],'-o')
    plt.xlabel('Iterations')
    plt.grid(True)


plt.figure(4,figsize=(20,10),dpi=200)
plt.suptitle('SVM')
plt.subplot(1,2,1)
plt.title('Training data')
plt.scatter(trainingDataClass0[:,0],trainingDataClass0[:,1],color='red')
plt.scatter(trainingDataClass1[:,0],trainingDataClass1[:,1],color='green')
plt.plot(separatingLineXcordinate,separatingLineYcordinate_svm,label='my svm')
plt.plot(separatingLineXcordinate,separatingLineYcordinate_svm_supportVec1,label='my svm SV1')
plt.plot(separatingLineXcordinate,separatingLineYcordinate_svm_supportVec2,label='my svm SV2')

# plt.plot(separatingLineXcordinate,separatingLineYcordinate_svmlib, label='sklearn SVM')
# plt.plot(separatingLineXcordinate,separatingLineYcordinate_svmlib_supportVec1, label='SV1')
# plt.plot(separatingLineXcordinate,separatingLineYcordinate_svmlib_supportVec2, label='SV2')
plt.legend()
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('y1')
plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])
# plt.axis('scaled')

plt.subplot(1,2,2)
plt.title('Training data')
plt.scatter(trainingDataClass0[:,0],trainingDataClass0[:,1],color='red')
plt.scatter(trainingDataClass1[:,0],trainingDataClass1[:,1],color='green')
plt.plot(separatingLineXcordinate,separatingLineYcordinate_svmlib, label='sklearn SVM')
plt.plot(separatingLineXcordinate,separatingLineYcordinate_svmlib_supportVec1, label='SV1')
plt.plot(separatingLineXcordinate,separatingLineYcordinate_svmlib_supportVec2, label='SV2')
plt.legend()
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('y1')
plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])
# plt.axis('scaled')

if 0:
    plt.subplot(1,2,2)
    plt.title('Testing data')
    plt.scatter(testingDataClass0[:,0],testingDataClass0[:,1],color='red')
    plt.scatter(testingDataClass1[:,0],testingDataClass1[:,1],color='green')
    plt.plot(separatingLineXcordinate,separatingLineYcordinate_svm,label='my SVM')
    # plt.plot(separatingLineXcordinate,separatingLineYcordinate_svmlib, label='sklearn SVM')
    # plt.plot(separatingLineXcordinate,separatingLineYcordinate_svmlib_supportVec2, label='SV1')
    # plt.plot(separatingLineXcordinate,separatingLineYcordinate_svmlib_supportVec2, label='SV2')
    plt.legend()
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.ylim([np.amin(Data[:,1])-2,np.amax(Data[:,1])+2])
    plt.xlim([np.amin(Data[:,0])-2,np.amax(Data[:,0])+2])


plt.figure(5,figsize=(20,10),dpi=200)
plt.title('SVM: Dual problem cost function vs iterations')
plt.plot(costFunctionDualProblem[1::],'-o')
plt.xlabel('Iterations')
plt.grid(True)




