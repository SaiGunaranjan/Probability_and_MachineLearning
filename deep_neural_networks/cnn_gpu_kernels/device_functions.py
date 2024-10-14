# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 23:47:50 2024

@author: Sai Gunaranjan
"""

from numba import cuda


@cuda.jit(device=True)
def convolution_2d_valid(input_matrix, kernel, result):
    """
    Performs 2D convolution in 'valid' mode in a C-style implementation.
    The kernel is flipped by 180 degrees manually.

    Parameters:
    - input_matrix: 2D list (or numpy array), input matrix.
    - kernel: 2D list (or numpy array), kernel matrix.

    Returns:
    - result: 2D list, the result of convolution.
    """
    # Get the input and kernel dimensions
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Output dimensions
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    # Manually flip the kernel (180 degrees)
    flipped_kernel = kernel[::-1,::-1]

    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            sum_value = 0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    sum_value += input_matrix[i + ki, j + kj] * flipped_kernel[ki,kj]
            result[i,j] = sum_value




@cuda.jit(device=True)
def convolution_2d_full(input_matrix, kernel, result, padded_input):
    """
    Performs 2D convolution in 'full' mode in a C-style implementation.
    Padding is added manually, and the kernel is flipped.

    Parameters:
    - input_matrix: 2D list (or numpy array), input matrix.
    - kernel: 2D list (or numpy array), kernel matrix.

    Returns:
    - result: 2D list, the result of convolution.
    """
    # Get the input and kernel dimensions
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    # Padding dimensions
    pad_height = kernel_height - 1
    pad_width = kernel_width - 1

    # Copy the original input matrix into the padded one
    for i in range(input_height):
        for j in range(input_width):
            padded_input[i + pad_height,j + pad_width] = input_matrix[i,j]

    # Call the 'valid' convolution on the padded input
    convolution_2d_valid(padded_input, kernel, result)
