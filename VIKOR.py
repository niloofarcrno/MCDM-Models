# Filename: VIKOR.py
# Description: VIKOR method
# Authors: Papathanasiou, J. & Ploskas, N.

from numpy import *
import matplotlib.pyplot as plt
import timeit
import os
import pandas as pd


# Step 1: determine the best and worst values for all
# criteria functions
def best_worst_fij(a, b):
    """ a is the array with the performances and b is
	the criteria min/max array 
	"""
    f = zeros((b.shape[0], 2))
    for i in range(b.shape[0]):
        if b[i] == 'max':
            f[i, 0] = a.max(0)[i]
            f[i, 1] = a.min(0)[i]
        elif b[i] == 'min':
            f[i, 0] = a.min(0)[i]
            f[i, 1] = a.max(0)[i]
    return f

# Step 2: compute the values S_i and R_i
def SR(a, b, c):
    """ a is the array with the performances, b is the
	array with the best and worst performances, and 
	c is the criteria min/max array 
	"""
    s = zeros(a.shape[0])
    r = zeros(a.shape[0])
    for i in range(a.shape[0]):
        k = 0
        o = 0
        for j in range(a.shape[1]):
            k = k + c[j] * (b[j, 0] - a[i, j]) \
                    / (b[j, 0] - b[j, 1])
            u = c[j] * (b[j, 0] - a[i, j]) \
                / (b[j, 0] - b[j, 1])
            if u > o:
                o = u
                r[i] = round(o, 3)
            else:
                r[i] = round(o, 3)
        s[i] = round(k, 3)
    return s, r

# Step 3: compute the values Q_i
def Q(s, r, n):
    """ s is the vector with the S_i values, r is
	the vector with the R_i values, and n is the
	number of criteria 
	"""
    q = zeros(s.shape[0])
    for i in range(s.shape[0]):
        q[i] = round((((n + 1) / (2 * n)) *
            (s[i] - min(s)) / (max(s) - min(s)) +
            (1 - (n + 1) / (2 * n)) *
            (r[i] - min(r)) / (max(r) - min(r))), 3)
    return q

# VIKOR method: it calls the other functions
def vikor(a, b, c, pl):
    """ a is the decision matrix, b is the criteria
	min/max array, c is the weights matrix, and pl 
	is 'y' for plotting the results or any other 
	string for not 
	"""
    s, r = SR(a, best_worst_fij(a, b), c)
    q = Q(s, r, len(c))
    if pl == 'y':
        e = [i + 1 for i in range(a.shape[0])]
        plt.plot(e, s, 'p--', color = 'red', 
			markeredgewidth = 2, markersize = 8)
        plt.plot(e, r, '*--',  color = 'blue', 
			markeredgewidth = 2, markersize=8)
        plt.plot(e, q, 'o--', color = 'green', 
			markeredgewidth = 2, markersize = 8)
        plt.legend(['S', 'R', 'Q'])
        plt.xticks(range(a.shape[0] + 2))
        plt.axis([0, a.shape[0] + 1, 0, 
			max(maximum(maximum(s, r), q)) + 0.3])
        plt.title("VIKOR results")
        plt.xlabel("Alternatives")
        plt.legend()
        plt.grid(True)
        plt.show()
    return s, r, q

# performances of the alternatives
x = array([[0.000180799, 61.65, 0.089409237, 1315.789474, 66.66666667, 0.028587764, 1.26963E-05, 13, 5, 7, 4]
       ,[0.000217202, 53.96, 0.102744421, 2040.816327, 103.0927835, 0.038387716, 1.86372E-05, 3, 1, 1, 6]
       ,[0.000168124, 65.28, 0.088787793, 1190.47619, 57.14285714, 0.026659557, 1.14426E-05, 14, 7, 5, 4]
       ,[0.000200401, 57.25, 0.099781325, 1851.851852, 100, 0.036218761, 1.69739E-05, 4, 3, 3, 7]])

# weights of the criteria
w = array([0.1089108911,0.1485148515, 0.08910891089, 0.1287128713, 0.06930693069, 0.06930693069, 0.05445544554, 0.06435643564, 0.1188118812, 0.07920792079,
0.06930693069])

# criteria max/min
crit_max_min = array(['max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max'])

# final results
start = timeit.default_timer()
vikor(x, crit_max_min, w, 'n')
stop = timeit.default_timer()
print(stop - start)
s, r, q = vikor(x, crit_max_min, w, 'y')
print("S = ", s)
print("R = ", r)
print("Q = ", q)
