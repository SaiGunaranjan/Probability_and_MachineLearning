# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 21:41:27 2025

@author: Sai Gunaranjan
"""

import numpy as np

n = 4  # Dimension of the matrix

# Step 1: Generate random orthogonal matrix Q
random_matrix = np.random.randn(n, n)
Q, _ = np.linalg.qr(random_matrix)

# Step 2: Create diagonal matrix D with values > 1
# eigenvalues = np.random.uniform(1.1, 5, size=n)  # smallest > 1

# Step 2: Create diagonal matrix D with values < 1
eigenvalues = np.random.uniform(0.1, 0.9, size=n)  # smallest > 1
D = np.diag(eigenvalues)

# Step 3: Construct symmetric matrix A
W = Q @ D @ Q.T

# Verify smallest eigenvalue
eigenVals = np.linalg.eigvalsh(W)
# print("Eigenvalues of A:", eigenVals)
# print("Smallest eigenvalue:", eigenVals.min())


numTimeSteps = 25

P = np.eye(n)
for i in range(numTimeSteps):

    # Generate random values
    random_values = np.random.randn(n)

    # Compute the sigmoid and its derivative
    sigmoid = 1 / (1 + np.exp(-random_values))
    derivatives = sigmoid * (1 - sigmoid)
    D_sigmoid_derivative = np.diag(derivatives)


    derivatives = 1 - np.tanh(random_values)**2 ## Compute the derivative of tanh function: 1 - tanh(x)^2
    D_tanh_derivative = np.diag(derivatives)


    P = P @ D_tanh_derivative @ W

    print('\nP at time step {} is \n {}'.format(i+1, P))