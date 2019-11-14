# Use the numpy library
import numpy as np


######################################################
#
#      MESSAGE TO STUDENTS:
#
#  This file contains a solution to the coding quiz. Feel free
#  to look at it when you are stuck, but try to solve the
#   problem on your own first.
#
######################################################


def prepare_inputs(inputs):
    # TODO: create a 2-dimensional ndarray from the given 1-dimensional list;
    #       assign it to input_array
    input_array = np.array([inputs])
    
    # TODO: find the minimum value in input_array and subtract that
    #       value from all the elements of input_array. Store the
    #       result in inputs_minus_min
    # We can use NumPy's min function and element-wise division
    inputs_minus_min = input_array - np.min(input_array)

    # TODO: find the maximum value in inputs_minus_min and divide
    #       all of the values in inputs_minus_min by the maximum value.
    #       Store the results in inputs_div_max.
    # We can use NumPy's max function and element-wise division
    inputs_div_max = inputs_minus_min / np.max(inputs_minus_min)

    return input_array, inputs_minus_min, inputs_div_max
    

def multiply_inputs(m1, m2):
    # Check the shapes of the matrices m1 and m2. 
    # m1 and m2 will be ndarray objects.
    #
    # Return False if the shapes cannot be used for matrix
    # multiplication. You may not use a transpose
    if m1.shape[0] != m2.shape[1] and m1.shape[1] != m2.shape[0]:     
        return False

    # Have not returned False, so calculate the matrix product
    # of m1 and m2 and return it. Do not use a transpose,
    #       but you swap their order if necessary
    if m1.shape[1] == m2.shape[0]:
        return np.matmul(m1, m2)        
    else:
        return np.matmul(m2, m1)        


def find_mean(values):
    # Return the average of the values in the given Python list
    # NumPy has a lot of helpful methods like this.
    return np.mean(values)
