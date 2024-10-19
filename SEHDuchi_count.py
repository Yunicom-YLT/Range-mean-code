import numpy as np
import random
from SEHDuchict import duchi


def outdomain(l,h,epsilon):
    e=np.exp(epsilon)
    y1=(((e+1)/(e-1))+1)*(h-l)/2+l
    y2=((-1*(e+1)/(e-1))+1)*(h-l)/2+l
    numbers = [y1,y2]
    return random.choice(numbers)





def duchi_count(data,lowerbound,higherbound,epsilon,frequency):
    e=np.exp(epsilon)
    y1 = (((e + 1) / (e - 1)) + 1) * (higherbound - lowerbound) / 2 + lowerbound
    y2 = ((-1 * (e + 1) / (e - 1)) + 1) * (higherbound - lowerbound) / 2 + lowerbound
    result=np.zeros(len(data))
    for i in range(len(data)):
        if (data[i] >= lowerbound) & (data[i] < higherbound):
            result[i]=duchi(data[i],lowerbound,higherbound,epsilon)
        else:
            result[i]=outdomain(lowerbound,higherbound,epsilon)

    return (np.sum(result)-len(data)*(1-frequency)*(y1+y2)/2)



# data=np.random.uniform(0,1000,100000)
# print(duchi_count(data,400,600,5,0.2))





