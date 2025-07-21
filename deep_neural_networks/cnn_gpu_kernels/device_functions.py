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
    # flipped_kernel = kernel # To bitmatch with tensorflow, I should not flip the kernel during convolution

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


@cuda.jit(device=True)
def max_val(array):
    max_value = float('-inf')  # Start with the smallest possible number

    height, width = array.shape

    # Iterate through the 2D array row by row
    for i in range(height):
        for j in range(width):
            # If the current element is greater than the current max, update max
            if array[i,j] > max_value:
                max_value = array[i,j]

    return max_value


@cuda.jit(device=True)
def argmax(array):
    max_value = float('-inf')  # Start with the smallest possible number
    max_index = -1  # To store the traveled index of the maximum element
    flattened_index = 0  # To keep track of the traveled (flattened) index

    height, width = array.shape

    # Iterate through the 2D array row by row
    for i in range(height):
        for j in range(width):
            # If the current element is greater than the current max, update max and index
            if array[i,j] > max_value:
                max_value = array[i,j]
                max_index = flattened_index

            # Increment traveled index after each element
            flattened_index += 1

    return max_index



@cuda.jit(device=True)
def max_pooling_devfunc(image, pool_size, stride, output, maxInd):
    """ This function borrowed from chat gpt. So verify this once"""
    """ Log the index of maxpool as well"""
    image_height, image_width = image.shape
    output_height = (image_height - pool_size) // stride + 1
    output_width = (image_width - pool_size) // stride + 1

    for y in range(0, output_height):
        for x in range(0, output_width):
            region = image[y*stride:y*stride+pool_size, x*stride:x*stride+pool_size]
            output[y, x] = max_val(region)
            maxInd[y, x] = argmax(region) # Needs to be unraveled


@cuda.jit(device=True)
def unravel_index(linear_index, num_rows, num_cols):
    # Calculate the row and column using integer division and modulo
    row = linear_index // num_cols
    col = linear_index % num_cols
    return (row, col)