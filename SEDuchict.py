import numpy as np
import random


def duchi_data(inputdata,epsilon):
    e=np.exp(epsilon)
    p=(e-1)*inputdata/(2*e+2)+0.5
    if random.random()<p:
        return (e+1)/(e-1)
    else:
        return -1*(e+1)/(e-1)

def duchi(data, min_val, max_val,epsilon):

    normalized_data = linear_normalize(data, min_val, max_val)
    normalized_data = duchi_data(normalized_data,epsilon)
    data_noisy = linear_denormalize(normalized_data, min_val, max_val)
    return np.sum(data_noisy)


def linear_normalize(array,min_val, max_val):
    normalized_array = 2*(array - min_val) / (max_val - min_val)-1
    return normalized_array


def linear_denormalize(normalized_array, min_val, max_val):
    denormalized_array = 0.5*(normalized_array+1) * (max_val - min_val) +min_val
    return denormalized_array

