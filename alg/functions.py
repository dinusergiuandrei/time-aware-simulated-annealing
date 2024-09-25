import numpy as np
import numba
from numba import jit


# add the value that obtains the minimum to measure distance to it
class Function:
    def __init__(self, name, f, real_minimum, real_argmin, lb, ub):
        self.name = name
        self.f = f
        self.real_minimum = real_minimum
        self.real_argmin = real_argmin
        self.lb = lb
        self.ub = ub


# part 1. slide 607. 9 functions
# functions from CEC 2013, 2019 are mentioned
#  1–3: Unimodal and valley shaped.
#  4–6: Multimodal. single global optimum. strong regularity.
#  7–9: multimodal. single global optimum. and high irregularity.
@jit(nopython=True) # 350 us -> 60 us
def sphere(x):
    return np.square(x).sum()

@jit(nopython=True) # 2.15 ms -> 70 us
def bohachevsky(x):
    x1 = x[:-1]
    x2 = x[1:]
    return np.sum(np.square(x1) + 2 * np.square(x2) - 0.3 * np.cos(3 * np.pi * x1) - 0.4 * np.cos(4 * np.pi * x2) + 0.7)

@jit(nopython=True) # 1.66 ms -> 50 us
def rosenbrock(x):
    x1 = x[:-1]
    x2 = x[1:]
    return np.sum(100 * np.square((x2 - np.square(x1))) + np.square(x1 - 1))

@jit(nopython=True) # 1.36 ms -> 65 us
def rastrigin(x):
    return 10 * len(x) + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))

@jit(nopython=True) # 3.11 ms -> 82 us. 38 times speed up
def ackley(x):
    return - 20 * np.exp(- 0.2 * np.sqrt(1 / len(x) * np.sum(np.power(x, 2)))) - \
           np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.exp(1)

@jit(nopython=True) # 1.81 ms -> 82 us
def griewangk(x):
    return np.sum(np.power(x, 2)) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

@jit(nopython=True) # 850 us -> 59 us
def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

@jit(nopython=True) # 1 ms -> 53 us
def eggholder(x):
    # 2 to 10
    # opts = np.array([-959.6406627, -1888.3213909, -2808.1847922, -3719.7248363,
    #         -4625.1447737, -5548.9775483, -6467.0193267, -7376.2797668, -8291.2400675])
    x1 = x[:-1]
    x2 = x[1:]
    s = np.sum(-(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) \
               - x1 * np.sin(np.sqrt(np.abs(x1 - x2 - 47))))
    # if 2 <= len(x) <= 10:
    #     s -= opts[len(x) - 2]
    # else:
    #     s -= - 915.61991 * len(x) + 862.10466
    return s

# https://arxiv.org/pdf/2003.09867.pdf
# −511.70430n + 511.68714
@jit(nopython=True) # 1.88 ms -> 57 us
def rana(x):
    # 2 to 7
    # opts = np.array([-511.7328819, -1023.4166105, -1535.1243381, -2046.8320657,
    #                 -2558.5397934, -3070.2475210])
    x1 = x[:-1]
    x2 = x[1:]
    s = np.sum(x1 * np.sin(np.sqrt(np.abs(x2 + 1 - x1))) * np.cos(np.sqrt(np.abs(x1 + x2 + 1))) + \
           (x2 + 1) * np.cos(np.sqrt(np.abs(x2 + 1 - x1))) * np.sin(np.sqrt(np.abs(x1 + x2 + 1))))
    # if 2 <= len(x) <= 7:
    #     s -= opts[len(x) - 2]
    # else:
    #     s -= - 511.70430 * len(x) + 511.68714
    return s


# ('Michalewicz', michalewicz, -1.8013, 0, np.pi),  # (2.2, 1.57)
# def michalewicz(x):
#     m = 10
#     t1 = np.sin(x)
#     t2 = np.sin([(i + 1) * xi ** 2 / np.pi for i, xi in enumerate(x)])
#     t3 = np.power(t2, 2 * m)
#     return - sum(t1 * t3)

functions_data = [
    ('Sphere', sphere, 0, [0, 0], -5.12, 5.12),
    ('Bohachevsky', bohachevsky, 0, [0, 0], -100, 100),
    ('Rosenbrock', rosenbrock, 0, [1, 1], -2.048, 2.048),
    ('Rastrigin', rastrigin, 0, [0, 0], -5.12, 5.12),
    ('Ackley', ackley, 0, [0, 0], -5.12, 5.12),
    ('Griewangk', griewangk, 0, [0, 0], -600, 600),
    ('Schwefel', schwefel, 0, [420.9687, 420.9687], -500, 500),
    ('Eggholder', eggholder, 0, [512, 404.2319], -512, 512),
    ('Rana', rana, 0, [-488.6326, 512], -512, 512),
]

functions = [Function(fd[0], fd[1], fd[2], fd[3], fd[4], fd[5]) for fd in functions_data]
# numba compilation
for _f in functions:
    _ = _f.f(np.random.uniform(-1, 1, 2))