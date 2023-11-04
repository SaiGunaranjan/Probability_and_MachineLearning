# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:30:10 2023

@author: Sai Gunaranjan
"""

"""
Kernel version of SVM

In this code, I have successfully implemented the kernel version of SVM. Specifically, I have used the
make_circles dataset to test the kernel SVM. This datasets has 2 classes as 2 circular rings(donut shape).
Since a normal SVM operates only on linearly separable datasets, we cannot directly apply the linear SVM.
The circular spread data has second order relationships and this can be captured by the 2nd order polynomial kernel
of the form (x_i.T x_j + 1)^p. I have implemented the kernel version of the SVM as a class as it ties up the methods
and data nicely. The results are looking good for this dataset. In the subsequenct commits, I will further test the
kernel SVM using the half moon datasets. For this, I cannot use a 2nd order poynomila kernel.
Since this has much higher order non-linear relationships across its features, I will use the gaussian kernel,
also called the radial basis function kernel. Kernel SVM is used when the original dataset is not linearly separable
but has some higher order relationships between its features such that through some tranformation of the vector
into a higher dimensional vector, the data then becomes linearly separable.
For example, if we have the make_circles data where the two classes of the dataset are separated by rings,
then we know that the relationship between the features is 2nd order. Now, by expanding the terms of the equation
of a circle, we can write it as an inner product of 2 vectors in a 6 dimensional space. Hence, the circular ring data
lies in a linear subspace in the 6 dimensional transformed subspace. So, in order to perform SVM on the ring dataset,
we first need to transform the vector from the 2 dimensional space to a 6 dimensional space through a
tranformation (say Phi) and then perform SVM in the 6 dimensional space. So then, we will find a separating hyperplane(W)
also in the 6 dimensional space. Also, once we perform the tranformation of the datapoints from 2D space to a 6D space,
we then need to perform pairwise dot products to get the xTx (in this case, after transformation it becomes
Phi(x)T phi(x)) which is required for evaluating:
1. cost function
2. wTx to obtain the class of the test data.

x is assumed to be of dimension #features x #datapoints in the theoretical formulation.
But in my simulation the data is of dimension #datapoints x #features

So, the compute is high because of these 2 operations:
1. Tranforming from low dimensional space to high dimensional space(size of the data increases)
2. We need to evaluate pairwise dot products (xTx) in the higher dimensional space, which is also compute intensive.

In the kernel SVM, we replace the operation xTx or xTy with the kernel matrix. The kernel matrix/function accomplishes
both the above 2 operations at one go. It takes care of handling non-linear/higher order relationships between the
features without explicitly tranforming to a higher dimensional space through what are know as kernel functions.
Kernel functions are functions which compute dot products in higher dimensional spaces without explicitly tranforming
the data to higher dimensional spaces. These are functions of the dot products in the lower dimenisonal spaces.
Ex: (x_iT * x_j + 1)^2 is a kernel function for 2nd order relationships like donut/moon datasets.
Kernel matrices satisfy 2 condtions:
1. Kernel matrices are symmetric. This is because, inner products are symmetric operations and hence the matrix
   is also symmetric.
2. They are positive semidefinite. This means that the eigen values of a kernel matrix are non-negative.
   This can be explained as follows. Kernel martices emulate the operation of pair wise dot products
   xTx (in higher dimensional space). Now, we know that the nonzero eigen values of xTx and xxT are same
   (this is a theorem and can be proved easily). But the eigen values of xxT are the variances of the projections
   of the data onto principal component vectors (or Eigen vectors of xxT). And variance is non-negative and hence
   the eigen value of xxT and hence xtx are non-negative. Hence the kernel matrix is a positive semi definite(PSD) matrix.
This is called the Mercer's theorem. This is a strong (if and only if condition) to check if a function is a valid kernel
or not.
Alternatively, if we can find a Phi in a high dimensional space such that Phi(x_i)T * Phi(x_j) = K(x_i,x_j),
then we can say that K is a valid kernel.

Implemented the RBF kernel for half moon datasets
Now, I have implemented the RBF(Radial Basis Function) kernel for data which have higher order non-linear relationships
between its features like half moon datasets, etc.
For linear relationships(across features), we can use the polynomial kernel of degree 1.
For 2nd order relationships, we can use the polynomial kernel of degree 2 (like circular datasets).
Similarly for higher order non-linear relationships, we can use the more generic gaussian/rdial basis function kernel.
This makes sense since e^x expansion has polynomials of infinite order/degree and hence can be use to model any kind of
non-linearities across the features of the datasets.
Some of the other changes made in this commit are:
    1. Passing the type of kernel to be used (for feature modelling) to the SVM initialization based on the type of dataset.
       For circular/donut datasets, we use the 2nd order polynomial kernel and for half moon datasets, we use the RBF kernel.
    2. Polynomial kernel is controlled by a variable named 'order' to define the order/degree of the polynomail kernel
       to be used. For now, since I'm using the donut dataset, I'm fixing the order to 2. But this can (and needs to) be
       changed based on the relationships between the features.
    3. Some code clean up.

Centering the kernel:
    When performing the transformation from low dimensional space to a higher dimensional space either explicitly
    or using a kernel function, the transformed features might not be centered around origin. In other words,
    transformed features may have a bias and this bias needs to be either removed or explicitly computed to get the
    true/correct separating hyperplane. If we are explicitly transforming the eatures to a high dimensioanl space,
    then we can remove the bias there before performing SVM. But if we are using kernels, we are not explicitly
    computing the tranformed features in the high dimensional space and hence cannot mean in the tranformed space.
    We are only computing the pair wise dot products in the high dimensional space using the kernel functions.
    In this case, we don't have access to the transformed features and hence we need to remove the bias in the kernel itself.
    This is called 'centering the kernel'. I have derived the closed form expression for centering the kernel.
    Here's the link to my derivation:
        https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/33423361/Kernel+centering
    There is another reference where the centering of the kernel is derived. Both mine and the reference are matching.
    Here's the link:
        https://twitter.com/gabrielpeyre/status/1708346002334466170?t=zYHl9vjj6NVCzA3yCDJZVQ&s=08
        https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
However, when I try to center the kernel in the code, the results are not satisfactory. Need to debug this.

Also, Now, I have a much better understanding of the penality term C imposed on the bribe. For a linearly separable
dataset, all the points satisfy the condition Wtxi*yi >= 1. But if the data set is not completely linearly separable
due to more noise in the data, then an ideal linear SVM(also called hard margin SVM) doesn't work. In such cases,
all points in the dataset do no satisfy the condition Wtxi*yi >= 1. So we might have some points violating this condition.
So, in order to make these points satisfy the separability constraint, we add a bribe to the points.
So the constraints now become Wtxi*yi + e_i >= 1, where e_i is the bribe paid for each point to satisfy the condition.
e_i (epsillon i) is a non-negative quantity. It is 0 for the points which already satisfy the condition Wtxi*yi >= 1 and
it is positive for points which don't satisfy. Now, by relaxing the constraints, even W = 0 can be made to pass the
linear separability constraints (by adding any number >1 as the epsillon). Since the goal of SVM is to find a W with
the least norm satisfying the linear separability constraints, then under the new relaxed condtions,
W=0 is the most suitable candidate since it has the the least norm (0) and also satisfies the linear separability
constraints and we get a trivial solution for the separatig hyperplane. So, it means, the problem statement is
not yet complete. What we are missing is that we have added bribes to relax the constraints and accomodate the
noisy points of datasets (which may not always be linearly separable) but we are not penalizing the bribes added.
So by adding this penalizer to the objective function and weighing it by say a parameter C, we can limit the bribes
paid by each point to satisfy the linear constraints. So now the objective function becomes:

min ||W||^2 / 2 + C * Sum(i=1, i=N)* e_i
W,e_i

s.t wTx_i * y_i + e_i >= 1 for all i,
e_i >= 0 for all i

This formulation is called the Soft margin SVM.

A larger C implies, we are heavily penalizing the bribes and hence we are forcing a very small/no bribe for the points
to satisy the linear separability condition. Hence, it like using a hard margin or the convention SVM with 0 bribes.
This means that if we have some noisy points close to the separating hyperplane wTx = 0, then we are essentially using
them as our support vectors. This means we are tring to fit for noise and this leads to overfitting. On the other hand,
if we make C too small, then we are not penalizing the bribes at all and hence almost all the points can pass the
linear constraints with arbitrarily choosing e_i. This will lead to a very relaxed SVM which does not accurately model
the separation between the 2 classes and this results in under fitting. So it is essential to carefully choose C through
cross validation on the training dataset.
More explanation of the hyper parameter C is available in the below link:
https://www.baeldung.com/cs/ml-svm-c-parameter#:~:text=Selecting%20the%20Optimal%20Value%20of%20C&text=depends%20on%20the%20specific%20problem,training%20error%20and%20margin%20width.

"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

plt.close('all')

class SVM:

    def __init__(self, trainingData, testingData, trainingLabels, kernelType):
        # testdata should be of dimension #datapoints x #features
        # trainingData is of shape #datapoints x #features
        self.trainingData = trainingData;
        self.testingData = testingData;
        self.trainingLabels = trainingLabels;
        self.numTrainingData = self.trainingData.shape[0]
        self.Y = np.diag(self.trainingLabels) # Diagonal matrix of class labels (+1/-1)

        """ Choice of hyper parameter C: Articles
        https://www.baeldung.com/cs/ml-svm-c-parameter#:~:text=Selecting%20the%20Optimal%20Value%20of%20C&text=depends%20on%20the%20specific%20problem,training%20error%20and%20margin%20width.
        """
        self.C = 10#10#0.05#40000

        """ Cap the maximum number of iterations of the gradient ascent step for solving the dual formulation
        variable alpha."""
        self.numMaxIterations = 1000#500

        self.kernelType = kernelType

        return

    def kernel_function(self,testdata):

        """
        Evaluates XTX or phi(X)Tphi(X) using kernel method
        """

        numEvalPoints = testdata.shape[0]
        if (self.kernelType == 'polynomial_kernel'):
            order = 2
            kernel = ((self.trainingData @ testdata.T) + 1)**order # Polynomial kernel (x_iTx_j + 1)^p
        elif (self.kernelType == 'radial_basis_function_kernel'):
            sigma = 0.1
            kernel = np.exp(-(np.linalg.norm(self.trainingData[:,:,None] - testdata.T[None,:,:], axis=1)**2)/(2*sigma**2)) # Radial basis function kernel e^(|| x - y||^2/(2sigma^2))

        """ Centering the kernel

        Here's the link to my derivation:
            https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/33423361/Kernel+centering
        There is another reference where the centering of the kernel is derived. Both mine and the reference are matching.
        Here's the link:
            https://twitter.com/gabrielpeyre/status/1708346002334466170?t=zYHl9vjj6NVCzA3yCDJZVQ&s=08
            https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        """
        self.oneVector_train = np.ones((self.numTrainingData,))
        oneVector_test = np.ones((numEvalPoints,))
        Iminus11T_train = np.eye(self.numTrainingData) - ((1/self.numTrainingData) * (self.oneVector_train[:,None] @ self.oneVector_train[None,:]))
        Iminus11T_test = np.eye(numEvalPoints) - ((1/numEvalPoints) * (oneVector_test[:,None] @ oneVector_test[None,:]))
        self.meanRemovedKernal = kernel#Iminus11T_train.T @ kernel @ Iminus11T_test # kernel

        return


    def svm_train(self):

        """ SVM training phase """



        """ The dataset might have a bias and need not always be around the origin. It could be shifted.
        For example, the data might be separated by the line wTx = -5. We do not know the bias apriori. So,
        to handle this, we do the following.
        We include the unknown bias also as another parameter to the w vector. We then also append a 1 to the feature vector
        ([x, 1]). Hence, the dimensionality of both the feature vector and the parameter vector w are increased by 1.
        """

        self.kernel_function(self.trainingData)

        # trainingDataExt = np.hstack((self.trainingData,np.ones((self.numTrainingData,1)))) # Appending 1s to the training data.

        self.alphaVec = np.zeros((self.numTrainingData,),dtype=np.float32)
        self.costFunctionDualProblem = np.zeros((self.numMaxIterations,),dtype=np.float32)
        for ele1 in range(self.numMaxIterations):
            self.costFunctionDualProblem[ele1] = (self.alphaVec @ self.oneVector_train) - (0.5 * (self.alphaVec[None,:] @ self.Y.T @ (self.meanRemovedKernal) @ self.Y @ self.alphaVec[:,None]))
            eta = 1/((ele1+1))#1e-4#1/((ele1+1)**2) # Learning rate/step size. Typically set as 1/t or 1/t**2
            gradient = self.oneVector_train - (self.Y.T @ (self.meanRemovedKernal) @ self.Y @ self.alphaVec) # Y.T @ (X.T @ X) @ Y can be moved outside the loop
            self.alphaVec = self.alphaVec + eta*gradient
            self.alphaVec[self.alphaVec<0] = 0 # box constraints. alpha should always be >=0
            self.alphaVec[self.alphaVec>self.C] = self.C
            # print('Min alpha val = {0:.5f}, Max alpha val = {1:.5f}'.format(np.amin(self.alphaVec),np.amax(self.alphaVec)))
        # Remove belowline
        # wVec_svm = X @ Y @ self.alphaVec # This is the actual wVec but it doesnt need to be explicitly computed
        suppVecIndMargin = np.where((self.alphaVec>0) & (self.alphaVec<self.C))[0]
        print('Number of supporting vectors = {}'.format(len(suppVecIndMargin)))

        """ Reference for the below method of bias calculation is given in:
            1. https://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-training-algorithms/#:~:text=Once%20training%20has%20been%20completed%2C%20we%20can%20calculate%20the%20bias%20b
            2. https://stats.stackexchange.com/questions/362046/how-is-bias-term-calculated-in-svms#:~:text=Generally%20speaking%20the%20bias%20term%20is%20calculated"""
        # bias = np.mean(trainingLabels[suppVecIndMargin] - (wVec_svm[0:-1][None,:] @ X[0:-1,suppVecIndMargin]))
        # bias = np.mean(trainingLabels[suppVecIndMargin] - (wVec_svm @ X[:,suppVecIndMargin]))
        # wVec_svm = np.hstack((wVec_svm,bias))
        """ There's some issue in the bias computation. Need to resolve this"""

        return



    def svm_test(self):

        """ Testing phase"""
        # numTestingData = self.testingData.shape[0]
        # testingDataExt = np.hstack((self.testingData,np.ones((numTestingData,1)))) # Appending 1s to the test data as well.
        wtx_test = self.decision_function(self.testingData)

        estLabels = np.zeros((wtx_test.shape),dtype=np.int32)
        estLabels[wtx_test>=0] = 1
        estLabels[wtx_test<0] = -1

        return estLabels


    def decision_function(self,Z):
        """ Generates wTz, where,
        w= wVec = X @ Y @ alpha,
        wVecT*Z = alpha.T @ Y.T @ XTZ = alpha.T @ Y.T @ K where K is the kernel matrix
        Z can be training data or test data"""

        self.kernel_function(Z)
        wtx_test = self.alphaVec[None,:] @ self.Y.T @ self.meanRemovedKernal
        wtx_test = wtx_test.squeeze()

        return wtx_test


    def decision_boundary(self):
        """ Plot the decision boundary/separating hyperplanes"""

        plt.scatter(self.trainingData[:, 0], self.trainingData[:, 1], c=self.trainingLabels, s=50, cmap=plt.cm.Paired, alpha=.5)
        ax = plt.gca()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        wtx = self.decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        CS = ax.contour(XX, YY, wtx, colors=['b', 'g', 'r'], levels=[-1, 0, 1], alpha=0.5,
                        linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])

        labels = ['WTx = -1', 'WTx = 0','WTx = 1']
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])

        plt.legend(loc='upper left')

        # highlight the support vectors
        ax.scatter(self.trainingData[:, 0][self.alphaVec > 0.], self.trainingData[:, 1][self.alphaVec > 0.], s=50,
                   linewidth=1, facecolors='none', edgecolors='k',label='support vectors (alpha > 0)')
        plt.legend()
        plt.grid(True)

    def svm_accuracy(self,testingLabels, estLabels):

        accuracy = np.mean(estLabels == testingLabels) * 100
        print('\nAccuracy of clasification SVM = {0:.2f} % \n'.format(accuracy))



# Data, labels = datasets.make_circles(n_samples=500, noise=0.05, random_state=None, factor=0.5)
# kernelType = 'polynomial_kernel'

Data, labels =  datasets.make_moons(n_samples=1000, noise=0.05, random_state=6)
kernelType = 'radial_basis_function_kernel'

numDataPoints = Data.shape[0]
numTrainingData = int(np.round(0.7 * numDataPoints))

trainingData = Data[0:numTrainingData,:]
trainingLabels = labels[0:numTrainingData]

trainingDataClass0 = trainingData[trainingLabels==0,:]
trainingDataClass1 = trainingData[trainingLabels==1,:]

testingData = Data[numTrainingData::,:]
testingLabels = labels[numTrainingData::]

testingDataClass0 = testingData[testingLabels==0,:]
testingDataClass1 = testingData[testingLabels==1,:]

""" Support Vector machine"""
trainingLabels_svm = trainingLabels.copy()
trainingLabels_svm[trainingLabels_svm==0] -= 1


testingLabels_svm = testingLabels.copy()
testingLabels_svm[testingLabels_svm==0] -= 1


""" Remove bias/mean in the data"""
# trainingData -= np.mean(trainingData,axis=0)[None,:]
# testingData -= np.mean(testingData,axis=0)[None,:]
svm_obj = SVM(trainingData, testingData, trainingLabels_svm, kernelType)
svm_obj.svm_train()
estLabels_svm = svm_obj.svm_test()
svm_obj.svm_accuracy(testingLabels_svm, estLabels_svm)
svm_obj.decision_boundary()


plt.figure(5,figsize=(20,10),dpi=200)
plt.title('SVM: Dual problem cost function vs iterations')
plt.plot(svm_obj.costFunctionDualProblem[1::],'-o')
plt.xlabel('Iterations')
plt.grid(True)