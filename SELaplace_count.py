import numpy as np
import random

def laplace_noisy(epsilon):
    n_value = np.random.laplace(0, 2/ epsilon, 1)
    return n_value


def laplace_mech(data, epsilon):
    data += laplace_noisy(epsilon)
    return data

def linear_normalize(array,min_val, max_val):
    normalized_array = (array - min_val) / (max_val - min_val)*2-1
    return normalized_array


def linear_denormalize(normalized_array, min_val, max_val):
    denormalized_array = 0.5*(normalized_array+1) * (max_val - min_val) + min_val
    return denormalized_array

def laplace_count(data, min_val, max_val,epsilon,frequency):
    result=np.zeros(len(data))
    for i in range(len(data)):
        if (data[i] >= min_val) & (data[i] < max_val):
           normalized_data = linear_normalize(data[i], min_val, max_val)
           data_noisy = laplace_mech(normalized_data, epsilon)
           if data_noisy>1:
               data_noisy=1
           if data_noisy<-1:
               data_noisy=-1
           result[i] = linear_denormalize(data_noisy, min_val, max_val)
        else:
            data_noisy=random.uniform(-1,1)
            result[i] = linear_denormalize(data_noisy, min_val, max_val)
    return (np.sum(result)-len(data)*(1-frequency)*(min_val+max_val)/2)

# data=np.random.uniform(0,1000,100000)
# print(laplace_count(data,400,600,5,0.2))
