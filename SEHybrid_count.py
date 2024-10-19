import numpy as np
from SEHDuchi_count import duchi_count
from SEHPiecewise_count import piecewise_count

def hybrid_count(data,lowerbound,higherbound,epsilon_pm,epsilon,freuqency):

    epsilon1=0.61
    if epsilon>epsilon1:
        perturb_p=1-np.exp(-epsilon/2)
        group_p = int(len(data) * perturb_p)

        # 使用choice函数将数据分为两组
        group_piecewise = np.random.choice(data, size=group_p , replace=False)
        group_duchi = np.random.choice(data, size=len(data)-group_p , replace=False)
        return np.add(piecewise_count(group_piecewise,lowerbound,higherbound,epsilon_pm,freuqency),duchi_count(group_duchi,lowerbound,higherbound,epsilon,freuqency))
        # return np.concatenate([opiecewise(group_piecewise,lowerbound,higherbound,epsilon), oduchi(group_duchi,lowerbound,higherbound,epsilon)])
    else:
        return duchi_count(data,lowerbound,higherbound,epsilon,freuqency)


# data=np.random.uniform(0,1000,100000)
# print(piecewise_count(data,400,600,5,0.2))


