import numpy as np
def se(data,epsilon):

    e=np.exp(epsilon)
    p=e/(e+1)
    data1=np.random.choice([0,1],size=np.count_nonzero(data== 0))
    data2 = np.random.choice([0, 1], size=np.count_nonzero(data == 1), p=[1 - p, p])
    data=np.concatenate([data1,data2])
    count1=np.count_nonzero(data == 1)
    true_count=(2*count1-len(data))/(2*p-1)
    frequency=(len(data)-true_count)/len(data)

    return frequency