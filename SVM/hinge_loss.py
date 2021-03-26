import numpy as np
import math

def margin(data, labels, th, th0):
    # data: d x n array
    # labels: 1 x n array
    # th: d x 1 array
    # th0: float or int
    # returns 1 x n array
    return labels * (np.dot(th.T, data) + th0) / math.sqrt(np.sum(th**2))

def hl_with_margin(data, labels, th, th0, yref):
    y = margin(data, labels, th, th0)           
    _, n = np.shape(y)
    l = np.zeros(n)
    for i in range(n):
        if y[0,i] < yref:
            l[i] = 1-y[0,i]/yref
        else: l[i] = 0
    return l 

