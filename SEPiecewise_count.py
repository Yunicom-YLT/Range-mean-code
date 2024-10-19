import random
import numpy as np
from SEPiecewisect import piecewise



def outdomain(l,h,epsilon):

    e2 = np.exp(epsilon/2)
    c=(e2+1)/(e2-1)
    y1=(c+1)*(h-l)/2+l
    y2=(-c+1)*(h-l)/2+l
    return random.uniform(y2,y1)


def piecewise_count(data,lowerbound,higherbound,epsilon,frequency):
    ee = np.exp(epsilon)
    e2 = np.exp(epsilon / 2)
    C = (e2 + 1) / (e2 - 1)
    p = (ee - e2) / (2 * e2 + 2)
    true = 1 / (4 * C * p)
    y1 = (C + 1) * (higherbound - lowerbound) / 2 + lowerbound
    y2 = (-C + 1) * (higherbound - lowerbound) / 2 + lowerbound
    result=np.zeros(len(data))
    for i in range(len(data)):
        if (data[i] >= lowerbound) & (data[i] < higherbound):
            result[i] = (piecewise(data[i], lowerbound, higherbound, epsilon))
        else:
            result[i]=outdomain(lowerbound,higherbound,epsilon)


    return (np.sum(result)-len(data)*(1-frequency)*(y1+y2)/2)

# data=np.random.uniform(0,1000,100000)
# print(piecewise_count(data,400,600,5,0.2))


