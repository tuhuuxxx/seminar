import numpy as np
from scipy.optimize import minimize
import math

points = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.8, 0.8]])
test = np.array([1, -3])
# x = [w1, w2, b, xi_1, xi_2, ..., xi_l] : l+3 - vector
def func(x):
    w1 = x[0]
    w2 = x[1]
    C = np.asarray([0.5 for i in range(3, x.size)])
    Xi = x[3:]
    return float((w1**2 + w2**2)/2 + np.sum(C*Xi))

l = points.shape[0]  # number of points 
A = np.c_[points, np.ones((l, 1)), np.identity(l)] # lx(l+3) matrix
b = [0 for i in range(l)] # l-vector


cons = [{"type": "ineq", "fun": lambda x: A @ x - b}, 
        {"type": "eq", "fun": lambda x: x[0]*test[0] + x[1]*test[1] + x[2] + 1}]

bnds = [(None, None), (None, None), (None, None)] # no boudaries for w1, w2, b
bnds = bnds + [(0, None) for i in range(l)]

xinit = [0.1 for i in range(l+3)]

sol = minimize(func, x0=xinit, bounds=bnds, constraints=cons)
w1, w2, b = sol.x[0:3]
def do(x, w1, w2, b):
    return abs(w1*x[0] + w2*x[1] + b)/(math.sqrt(w1**2 + w2**2))
print(do(test, w1, w2, b))
