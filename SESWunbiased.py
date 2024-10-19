import numpy as np


def sw_unbiased(ori_samples, l, h, eps):

    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2 #选择的2b
    p = ee / (w * ee + 1)#高概率p
    q = 1 / (w * ee + 1)#低概率q

    samples = (ori_samples - l) / (h - l)
    randoms = np.random.uniform(0, 1, len(samples))

    noisy_samples = np.zeros_like(samples)

    # report
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2
    

    b=w/2
    k = q * (b + 0.5)
    c = noisy_samples - k
    final = c / (2 * b * (p - q))
    final = final * (h - l) + l

    return final


def sw_count(data,lowerbound,higherbound,epsilon,frequency):
    ee = np.exp(epsilon)
    b = ((epsilon * ee) - ee + 1) / (2 * ee * (ee - 1 - epsilon))
    ho=(1+b)*(higherbound-lowerbound)+lowerbound
    lo=(-b)*(higherbound-lowerbound)+lowerbound
    data_in=data[(data>=lowerbound)&(data<=higherbound)]
    data_out=np.random.uniform(lo,ho,len(data)-len(data_in))

    data_in=sw_unbiased(data_in, lowerbound,higherbound,epsilon)

    output_sw=np.concatenate([data_in,data_out])

    return (np.sum(output_sw)-len(data)*(1-frequency)*(ho+lo)/2)


# data=np.random.uniform(0,1000,100000)
# print(sw_count(data,400,600,5,0.5))