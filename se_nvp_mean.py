import random
from se import se
import numpy as np
from SELaplace_count import laplace_count
from SEHybrid_count import hybrid_count
from SEDuchi_count import duchi_count
from SEPiecewise_count import piecewise_count
from SESWunbiased import sw_count
from SEara_orrnvp import ara_orr
import math
import sympy
from sympy import *

def solve_lm(epsilon):
    d = sympy.Symbol("d")
    k=np.exp(epsilon)
    f=-k + 0.5 * sympy.exp(d/2) * (1 + sympy.exp(d))
    b1 = sympy.nsolve(f, d, (0.5*epsilon,epsilon),tol=1e-6)

    a = sympy.Symbol("a")
    k=np.exp(epsilon)
    f1=-k + 0.25 * (sympy.exp(a/2) * (1 + sympy.exp(a)))*a/(-1+sympy.exp(a/2)+1e-10)
    b2 = sympy.nsolve(f1, a, (0.5*epsilon,epsilon),tol=1e-6)

    c = sympy.Symbol("c")
    k=np.exp(epsilon)
    f2=-k +4*sympy.exp(3*c/2)*(-1+sympy.exp(c/2))/(1+sympy.exp(c))/(c+1e-10)
    b3 = sympy.nsolve(f2, c, (0.5*epsilon,epsilon),tol=1e-6)

    return min(b1,b2,b3)
def solve_ph(epsilon):
    d = sympy.Symbol("d")
    k=np.exp(epsilon)
    f=-k + 0.5 * sympy.exp(d/2) * (1 + sympy.exp(d))
    b = sympy.nsolve(f, d, (0.5*epsilon,epsilon),tol=1e-6)
    return b

def solve_sw(epsilon):
    d = sympy.Symbol("d")
    k=np.exp(epsilon)
    f=-k + (sympy.exp(2*d)-1)/(2*d+1e-100)
    b = sympy.nsolve(f, d, (0.5*epsilon,epsilon),tol=1e-6)

    return b


def se_nvp(data,l,h,epsilon,ranges):
    epsilon_ph=float(solve_ph(epsilon))
    epsilon_sw=float(solve_sw(epsilon))
    epsilon_lm=float(solve_lm(epsilon))

    is_in_range = np.logical_and(data >= ranges[0], data <= ranges[1])# 生成新的数组，范围内的元素为0，不在范围内的元素为1
    krr_data = np.where(is_in_range, 0, 1)
    frequency=se(krr_data,epsilon)#第一个是范围内的频率，第二个是范围外的频率
    frequency_ph = se(krr_data, epsilon_ph)
    frequency_sw=se(krr_data,epsilon_sw)
    frequency_lm=se(krr_data,epsilon_lm)

    krr_lm=laplace_count(data, ranges[0],ranges[1],epsilon_lm,frequency_lm)/frequency_lm/len(data)
    krr_sr=duchi_count(data,ranges[0],ranges[1],epsilon,frequency)/frequency/len(data)
    krr_pm = piecewise_count(data, ranges[0], ranges[1], epsilon_ph, frequency_ph)/frequency_ph/len(data)
    if epsilon<0.61:
        krr_hm=duchi_count(data,ranges[0],ranges[1],epsilon,frequency)/frequency/len(data)
    else:
        krr_hm = hybrid_count(data, ranges[0], ranges[1], epsilon_ph,epsilon, frequency_ph)/frequency_ph/len(data)
    krr_sw = sw_count(data, ranges[0], ranges[1], epsilon_sw, frequency_sw) / frequency_sw / len(data)

    return krr_lm,krr_sr,krr_pm,krr_hm,krr_sw


def se_nvp_ara(data,l,h,epsilon,ranges):

    epsilon_ph=float(solve_ph(epsilon))
    epsilon_sw=float(solve_sw(epsilon))
    epsilon_lm=float(solve_lm(epsilon))

    shuffled_array = np.random.permutation(data)
    # 计算10%的索引位置
    split_index = int(0.1 * len(shuffled_array))

    # 分割数组
    datasw = shuffled_array[:split_index]
    datanvp = shuffled_array[split_index:]

    denselm, densesr, densepm, densehm, densesw, ratelm, ratesr, ratepm, ratehm, ratesw, exvlm, exvsr, exvpm, exvhm, exvsw \
        = ara_orr(datasw, l, h, epsilon, ranges, len(shuffled_array))

    print("seara:", denselm, densesr, densepm, densehm, densesw)

    is_in_rangelm = np.logical_and(datanvp >= ranges[0], datanvp <= ranges[1])  # 生成新的数组，范围内的元素为0，不在范围内的元素为1
    krr_datalm = np.where(is_in_rangelm, 0, 1)
    frequencylm = se(krr_datalm, epsilon_lm)  # 第一个是范围内的频率，第二个是范围外的频率

    is_in_rangesr = np.logical_and(datanvp >= ranges[0], datanvp <= ranges[1])  # 生成新的数组，范围内的元素为0，不在范围内的元素为1
    krr_datasr = np.where(is_in_rangesr, 0, 1)
    frequencysr = se(krr_datasr, epsilon)  # 第一个是范围内的频率，第二个是范围外的频率

    is_in_rangepm = np.logical_and(datanvp >= ranges[0], datanvp <= ranges[1])  # 生成新的数组，范围内的元素为0，不在范围内的元素为1
    krr_datapm = np.where(is_in_rangepm, 0, 1)
    frequencypm = se(krr_datapm, epsilon_ph)  # 第一个是范围内的频率，第二个是范围外的频率

    is_in_rangehm = np.logical_and(datanvp >= ranges[0], datanvp <= ranges[1])  # 生成新的数组，范围内的元素为0，不在范围内的元素为1
    krr_datahm = np.where(is_in_rangehm, 0, 1)
    if epsilon<0.61:
        frequencyhm = se(krr_datahm, epsilon)
    else:
        frequencyhm = se(krr_datahm, epsilon_ph)

    is_in_rangesw = np.logical_and(datanvp >= ranges[0], datanvp <= ranges[1])  # 生成新的数组，范围内的元素为0，不在范围内的元素为1
    krr_datasw = np.where(is_in_rangesw, 0, 1)
    frequencysw = se(krr_datasw,  epsilon_sw)  # 第一个是范围内的频率，第二个是范围外的频率

    krr_laplace = (laplace_count(datanvp, denselm[0], denselm[1], epsilon_lm, frequencylm) * 10 / 9 ) / (
            frequencylm) / len(data)
    krr_sr = (duchi_count(datanvp, densesr[0], densesr[1], epsilon, frequencysr) * 10 / 9 ) / (
            frequencysr ) / len(data)
    krr_pm = (piecewise_count(datanvp, densepm[0], densepm[1], epsilon_ph, frequencypm) * 10 / 9 ) / (
            frequencypm ) / len(data)
    krr_sw = (sw_count(datanvp, densesw[0], densesw[1], epsilon_sw, frequencysw) * 10 / 9 ) / (
            frequencysw ) / len(data)

    if epsilon<0.61:
        krr_hm=(duchi_count(datanvp, densesr[0], densesr[1], epsilon, frequencysr) * 10 / 9 ) / (
            frequencysr ) / len(data)
    else:
        krr_hm =(hybrid_count(datanvp, densehm[0],densehm[1], epsilon_ph,epsilon, frequencypm,) * 10 / 9 ) / (
                frequencypm ) / len(data)
    return krr_laplace, krr_sr, krr_pm, krr_hm, krr_sw


# data=np.random.uniform(0,1000,1000000)
# print(se_nvp_mean_ara(data,0,1000,0.5,(250,750)))
# print(solve_lm(0.05),solve_sw(0.05),solve_ph(0.05))