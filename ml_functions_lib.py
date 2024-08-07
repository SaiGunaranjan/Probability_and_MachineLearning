# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 22:59:03 2023

@author: Sai Gunaranjan
"""

import numpy as np
from scipy.special import expit as sigmoid
from scipy import special

"""
Naive Bayes is a generative machine learning model.
Perceptron, Logistic regression, k-nearest neighbours, decision trees are discriminative ML models.
"""

""" Good reference for perceptron implementation:
    https://towardsdatascience.com/perceptron-explanation-implementation-and-a-visual-example-3c8e76b4e2d1

Perceptron algorithm

In this script, I have implemented the perceptron algorithm. Perceptron algorithm is a supervised learning algorithm for
 binary classification when the data is linearly separable with a "gamma margin separability". The implementation is based on the video lectures of Arun Rajkumar and
the blog:
https://towardsdatascience.com/perceptron-explanation-implementation-and-a-visual-example-3c8e76b4e2d1

For generating the 2 class data/labels, I have used the "make_classification" function of the datasets class imported from
the sklearn package. Here we can mention the number of data samples, dimensionality of the feature vector, number of classes,
number of clusters per class, amount of separation across the classes and so on. This is a very useful function for data generation.
The perceptron algorithm converges when the training data is perfectly linearly separable with a gamma margin i.e
wTx must not just be greater than zero but is should be greater than 0 by a gamma margin. In other words,
the perceptron algorithm converges if wTx*y >= gamma , for all x and a positive gamma. So,
linear separability is only a necessary condition but not a sufficient condition. But when the training data is mixed
i.e. not perfectly linearly separable, then the perceptron algorithm does not converge. This can be theoretically proved and
the proof is given in the lectures of Arun Rajkumar.  But with actual real data, we cannot always ensure that the training data
is linearly separable (or with a gamma margin). Hence, the perceptron algorithm might not converge at all.
So in order to implement the perceptron algorithm even in such cases, we need to cap the maximum number
of iterations that the perceptron algorithm runs for, else it will never converge. So, even in the code, I have fixed
the maximum number of iterations for the percpetron algorithm. Theoretically, we can show that the time/number of iterations for
the perceptron algorithm to converge is proportional to the radius of the farthest data point and inversely proportional to
the gamma separation/margin between the closest points between the two classes. The closer the points of the two classes,
the more time the percpetron algorithm takes to converge.

"""

def perceptron_train(trainingData,trainingLabels):

    """ Training labels should be +1/-1"""

    """ Perceptron training phase """

    """ Cap the maximum number of iterations of the perceptron algorithm. Ideally this should be a function of radius of
    farthest point in the dataset and the gamma margin of separation between the 2 classes."""
    numMaxIterations = 100

    numTrainingData = trainingData.shape[0]
    numFeatures = trainingData.shape[1]

    """ The dataset might have a bias and need not always be around the origin. It could be shifted.
    For example, the data might be separated by the line wTx = -5. But the perceptron algorithm is defined assuming data
    doesn't have any bias. So, just checking for sign(wTx) to get the class labels will not be correct. We need to check if wTx >= -5 or wTx <-5.
    Also, we do not know the bias apriori. So, to handle this, we do the following.
    We include the unknown bias also as another parameter to the w vector. We then also append a 1 to the feature vector ([x, 1]). Hence,
    the dimensionality of both the feature vector and the parameter vector w are increased by 1.
    """
    wVec = np.zeros((numFeatures+1,),dtype=np.float32) # +1 to take care of the bias in the data
    alpha = 1 # Added a momentum parameter but set it to 1 for now.
    for ele1 in range(numMaxIterations):
        for ele2 in range(numTrainingData):
            xVec = trainingData[ele2,:]
            xVecExt = np.hstack((xVec,1)) # Appending 1 to the feature vector as well.
            wtx = np.sum(wVec * xVecExt)
            if (wtx*trainingLabels[ele2]) <= 0: # The equal sign is to handle the initialization of w to 0 and hence an update is required at the first iteration itself.
                wVec = wVec + alpha*(xVecExt * trainingLabels[ele2]) # Update equation for the perceptron algorithm : W_k+1 = W_k + x*y

    return wVec

def perceptron_test(testingData,wVec):

    """ Testing phase"""
    numTestingData = testingData.shape[0]
    testingDataExt = np.hstack((testingData,np.ones((numTestingData,1)))) # Appending 1s to the test data as well.

    wtx_test = testingDataExt @ wVec
    estLabels = np.zeros((wtx_test.shape),dtype=np.int32)
    estLabels[wtx_test>=0] = 1
    estLabels[wtx_test<0] = -1

    return estLabels

def perceptron_accuracy(testingLabels, estLabels):

    accuracy = np.mean(estLabels == testingLabels) * 100
    print('Accuracy of clasification for Perceptron = {0:.2f} % '.format(accuracy))



"""
Logistic algorithm

In this script, I have implemented the Logistic regression algorithm. Logistic regression is a
supervised learning algorithm for binary classification when the data is linearly separable to some extent.
It doesnt require a strict constraint on the linear separability. Even though it is called logistic "regression",
it is not a regression algorithm. It is a classification algorithm. But it is called regression because,
in some sense we are regressing the parameter vector W.
The implementation is based on the video lectures of Arun Rajkumar.


For generating the 2 class data/labels, I have used the "make_classification" function of the datasets class imported from
the sklearn package. Here we can mention the number of data samples, dimensionality of the feature vector, number of classes,
number of clusters per class, amount of separation across the classes and so on. This is a very useful function for data generation.


The perceptron algorithm is a deterministic linear classifier i.e P(Y=1/X) = 1 if wTx >= 0 and 0 otherwise.
Also, it works well if the data is linearly separable with a gamma margin. However, the logistic regression algorithm
gives a probabilistic measure for each data point and hence the data need not be strictly linearly separable.
The dot product of the parameter vector W with the data point is used a weight for that data point. So,
the more positve the dot product wTx, greater the probability that the point belongs to class 1. Similarly,
the more negative wTx, lesser the probability that the point belongs to label 1 (or equivalently, greater the probability
the data point belongs to class 0). If the dot product wTx = 0, then it is equally likely to come from class 1,0.
So, now we need to convert these scores for each of the data points to a probabilistic measure.
The probabisitic measure/function mapping (from wTx to probability) should be as follows:
    1. P(Y=1/X) --> 1 as wTx --> infinity
    2. P(Y=1/X) --> 0 as wTx --> -infinity ( or P(Y=0/X) --> 1)
    3. P(Y=1/X) = 0.5 if wTx = 0
One of the function which satisfies this mapping from wTx (-infinity, infinity) to [0,1] is the
logistic/sigmoid function. It is given by 1 / (1 + e**(-wTx)).
With this picture in mind, we can think of each data point being generated by a coin toss whose probability of
head i.e P(Y=1/X) is 1 / (1 + e**(-wTx)). So, more positive the inner product wTx, more likely it is to come from class 1
(It still has a finite non-zero probability of coming from class 0). Similarly, the more negative wTx, the more likely
the data point belongs to class 0 (here also it has a finite non-zero probability of belonging to class 1).
Thus the logistic regression algorithm accomodates non-linearly separable datasets as well.
The update rule for the logistic regression algo is based on the maximizing the log-likelihood function.
But due to the complex form of the sigmoid function, we do not get a close form expression for the maximizer W.
Instead we find the gradient and update the W at each iteration. The derivation is clearly explained in Arun's video lectures.
The update equation for W is a linear combination of the data points each weighted by the difference of the true label
and the estimated label (as given by the sigmoid function value for that data point). The update equation is as follows:
    W(k+1) = W(k) + alpha * summation (Xi (Yi-P(Y=1/Xi))), where P(Y=1/Xi) is the sigmoid function.
From this equation, we can clearly see that the data points for which the true labels and the estimated labels (rom sigmoid function)
match, do not contribute to the W update. It is those points for which there is a mismatch between the true labels
and estimated labels, that contribute to the W update. This is key observation which lays the foundation for
more sophisticated algos like SVms, etc.

Note:
    In perceptron algorithm, we treat the labels as +1, -1.
    In logistic regression algorithm, we treat the labels as 1, 0
The above assumptions just make the math easier.

Below blog has a good explanation for the logistic regression algorithm:
    https://mlu-explain.github.io/logistic-regression/

There are other variants of the logostic regression called bayesian logistic regression, etc. which I will try to cover gradually.
The bayesian regression logistic regression assumes a bayesian prior on the distribution of the W vector
i.e. the elements of the W vector are assumed to be drawn from a distribution say Gaussian, etc.
I will study this in more detail and then implement this.

"""
def logistic_regression_train(trainingData,trainingLabels):

    """ Logistic Regression training phase """

    """ Cap the maximum number of iterations of the Logistic regression algorithm."""
    numMaxIterations = 100

    numTrainingData = trainingData.shape[0]
    numFeatures = trainingData.shape[1]

    """ The dataset might have a bias and need not always be around the origin. It could be shifted.
    For example, the data might be separated by the line wTx = -5. We do not know the bias apriori. So,
    to handle this, we do the following.
    We include the unknown bias also as another parameter to the w vector. We then also append a 1 to the feature vector
    ([x, 1]). Hence, the dimensionality of both the feature vector and the parameter vector w are increased by 1.
    """
    wVec = np.zeros((numFeatures+1,),dtype=np.float32) # +1 to take care of the bias in the data

    trainingDataExt = np.hstack((trainingData,np.ones((numTrainingData,1)))) # Appending 1s to the training data.
    logLikelihood = np.zeros((numMaxIterations,),dtype=np.float32)
    for ele1 in range(numMaxIterations):
        alpha = 1/(ele1+1) # Learning rate/step size. Typically set as 1/t or 1/t**2
        wTx = trainingDataExt @ wVec

        # p_Yequal1_condX = 1 / (1 + np.exp(-wTx))
        # logLikelihood[ele1] = np.sum(-wTx + wTx*trainingLabels - np.log(1+np.exp(-wTx)))
        """ The above piece of code sufferes from overflows. This is because, when the dot product wTx becomes very large
        (both positive and negative), it can cause over flow while evaluating exp(-wTx) and also log(1+exp(-wTx)).
        This will result in NaNs. To avoid this, we use the inbuilt sigmoid function and the special.logsumexp function
        which evaluates log(sum of powers of e). We can pass the powers of e as an input vector to this function.
        The overflows are gracefully handled by these functions.
        The references for these issues and fixes is available in the below links:
            https://fa.bianp.net/drafts/derivatives_logistic.html

        """
        p_Yequal1_condX = sigmoid(wTx)
        temp1 = np.zeros((numTrainingData))
        temp2 = np.hstack((temp1[:,None],-wTx[:,None]))
        logLikelihood[ele1] = np.sum(-wTx + wTx*trainingLabels - special.logsumexp(temp2,axis=1)) #np.sum(-wTx + wTx*trainingLabels - np.log(sigmoid(wTx)))
        wVec_Kminus1 = wVec
        wVec = wVec + alpha*np.sum(trainingDataExt * (trainingLabels[:,None] - p_Yequal1_condX[:,None]),axis=0)
        fractionalErrorNorm = np.linalg.norm(wVec - wVec_Kminus1) / np.linalg.norm(wVec)
        # print('Iter {0:d}, fractional error(dB) {1:.2f}'.format(ele1, 10*np.log10(np.abs(fractionalErrorNorm))))

    return wVec, logLikelihood


def logistic_regression_test(testingData,wVec):

    """ Testing phase"""
    numTestingData = testingData.shape[0]
    testingDataExt = np.hstack((testingData,np.ones((numTestingData,1)))) # Appending 1s to the test data as well.

    wtx_test = testingDataExt @ wVec
    estLabels = np.zeros((wtx_test.shape),dtype=np.int32)
    estLabels[wtx_test>=0] = 1
    estLabels[wtx_test<0] = 0

    return estLabels


def logistic_regression_accuracy(testingLabels, estLabels):

    accuracy = np.mean(estLabels == testingLabels) * 100
    print('\nAccuracy of clasification Logistic regression = {0:.2f} % \n'.format(accuracy))


"""

I have implemented the soft margin SVM algorithm based on the video lectures of Arun Rajkumar.
However, currently, my implementation of the SVM is not giving the same supporting hyperplanes as the
sklearn SVM. I need to debug this. Also, there is another confusion regarding the estimation of bias term.
Do we add a "1" to the feature vector and estimate the bias or do we explicitly estimate?
I need to understand this as well.

The derivation of the hard margin SVM based on Arun's video lectures is available here:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/28114945/Support+Vector+Machine

SVM good reference article:
https://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-training-algorithms/
"""

def svm_train(trainingData,trainingLabels):

    """ SVM training phase """

    """ Cap the maximum number of iterations of the gradient ascent step for solving the dual formulation
    variable alpha."""
    numMaxIterations = 1000#500

    numTrainingData = trainingData.shape[0]
    # numFeatures = trainingData.shape[1]

    """ The dataset might have a bias and need not always be around the origin. It could be shifted.
    For example, the data might be separated by the line wTx = -5. We do not know the bias apriori. So,
    to handle this, we do the following.
    We include the unknown bias also as another parameter to the w vector. We then also append a 1 to the feature vector
    ([x, 1]). Hence, the dimensionality of both the feature vector and the parameter vector w are increased by 1.
    """

    Y = np.diag(trainingLabels)
    oneVector = np.ones((numTrainingData,))
    trainingDataExt = np.hstack((trainingData,np.ones((numTrainingData,1)))) # Appending 1s to the training data.
    X = trainingDataExt.T
    # X = trainingData.T
    """ Choice of hyper parameter c: Articles
    https://www.baeldung.com/cs/ml-svm-c-parameter#:~:text=Selecting%20the%20Optimal%20Value%20of%20C&text=depends%20on%20the%20specific%20problem,training%20error%20and%20margin%20width.
    """
    c = 10#0.05#40000
    alphaVec = np.zeros((numTrainingData,),dtype=np.float32)
    # alphaVec = np.random.random(numTrainingData)
    costFunctionDualProblem = np.zeros((numMaxIterations,),dtype=np.float32)
    for ele1 in range(numMaxIterations):
        costFunctionDualProblem[ele1] = (alphaVec @ oneVector) - (0.5 * (alphaVec[None,:] @ Y.T @ (X.T @ X) @ Y @ alphaVec[:,None]))
        eta = 1e-4#1/((ele1+1)**2) # Learning rate/step size. Typically set as 1/t or 1/t**2
        gradient = oneVector - (Y.T @ (X.T @ X) @ Y @ alphaVec) # Y.T @ (X.T @ X) @ Y can be moved outside the loop
        alphaVec = alphaVec + eta*gradient
        alphaVec[alphaVec<0] = 0 # box constraints. alpha should always be >=0
        alphaVec[alphaVec>c] = c
        # print('Min alpha val = {0:.5f}, Max alpha val = {1:.5f}'.format(np.amin(alphaVec),np.amax(alphaVec)))

    wVec_svm = X @ Y @ alphaVec # This is the actual wVec but it doesnt need to be explicitly computed
    suppVecIndMargin = np.where((alphaVec>0) & (alphaVec<c))[0]
    print('Number of supporting vectors = {}'.format(len(suppVecIndMargin)))

    """ Reference for the below method of bias calculation is given in:
        1. https://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-training-algorithms/#:~:text=Once%20training%20has%20been%20completed%2C%20we%20can%20calculate%20the%20bias%20b
        2. https://stats.stackexchange.com/questions/362046/how-is-bias-term-calculated-in-svms#:~:text=Generally%20speaking%20the%20bias%20term%20is%20calculated"""
    # bias = np.mean(trainingLabels[suppVecIndMargin] - (wVec_svm[0:-1][None,:] @ X[0:-1,suppVecIndMargin]))
    # bias = np.mean(trainingLabels[suppVecIndMargin] - (wVec_svm @ X[:,suppVecIndMargin]))
    # wVec_svm = np.hstack((wVec_svm,bias))
    """ There's some issue in the bias computation. Need to resolve this"""

    return wVec_svm, costFunctionDualProblem


def svm_test(testingData,wVec_svm):

    """ Testing phase"""
    numTestingData = testingData.shape[0]
    testingDataExt = np.hstack((testingData,np.ones((numTestingData,1)))) # Appending 1s to the test data as well.

    wtx_test = testingDataExt @ wVec_svm
    # wtx_test = testingData @ wVec_svm
    estLabels = np.zeros((wtx_test.shape),dtype=np.int32)
    estLabels[wtx_test>=0] = 1
    estLabels[wtx_test<0] = -1

    return estLabels


def svm_accuracy(testingLabels, estLabels):

    accuracy = np.mean(estLabels == testingLabels) * 100
    print('\nAccuracy of clasification SVM = {0:.2f} % \n'.format(accuracy))
