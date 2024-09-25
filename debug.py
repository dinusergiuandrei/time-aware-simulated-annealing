import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from alg.functions import functions
from alg.bits import bit_metrics, \
    bits_to_value, value_to_bits, \
    value_to_bits_v2
from alg.sa import sa, acc_prob_ln
from time import time


def sa_test():
    root = os.path.join('experiments', 'experiment_1_ln_vs_exp')

    target_function = functions[0]

    scores, accept_probs, outcomes = sa(target_function=target_function.f,
                                        lb=target_function.lb, ub=target_function.ub,
                                        acc_prob_=acc_prob_ln, T0=500, apf_param=1e-3, p_max=0.8,
                                        n_dims=int(2), max_iter=100)
    print(scores)


def list_test(xs, n_bits, lb, ub, max_on_bits):
    ts = np.empty(shape=(xs.shape[0], n_bits), dtype=np.float64)
    for i, x in enumerate(xs):
        ts[i] = value_to_bits(x, n_bits, lb, ub, max_on_bits)
    return ts


def array_test(xs, n_bits, lb, ub, max_on_bits):
    ts = np.empty(shape=(xs.shape[0], n_bits), dtype=np.float64)
    for i, x in enumerate(xs):
        ts[i] = value_to_bits_v2(x, n_bits, lb, ub, max_on_bits)
    return ts

def time_test():
    lb = 0
    ub = 8
    tolerance = 1e-7
    n_bits, powers, max_value_on_bits = bit_metrics(lb, ub, tolerance)

    xs = np.random.uniform(lb, ub, size=3)
    _ = list_test(xs, n_bits, lb, ub, max_value_on_bits)
    _ = array_test(xs, n_bits, lb, ub, max_value_on_bits)

    xs = np.random.uniform(lb, ub, size=6)

    start_time = time()
    t1 = list_test(xs, n_bits, lb, ub, max_value_on_bits)
    list_time = time() - start_time

    start_time = time()
    t2 = array_test(xs, n_bits, lb, ub, max_value_on_bits)
    array_time = time() - start_time

    v1 = []
    for t in t1:
        v1.append(bits_to_value(t, lb, ub, max_value_on_bits, powers))
    v1 = np.array(v1)

    v2 = []
    for t in t1:
        v2.append(bits_to_value(t, lb, ub, max_value_on_bits, powers))
    v2 = np.array(v2)

    print('List time: ', list_time)
    print('Array time: ', array_time)
    print('Max diff: ', np.max(np.abs(t1 - t2)))
    print(xs)
    print(v1)
    print(v2)
    print('List err: ', np.max(np.abs(xs - v1)))
    print('Array err: ', np.max(np.abs(xs - v2)))


if __name__ == '__main__':
    # xs = np.array([0, 0.4, 0.5, 0.8, 0.9, 1])
    # ys = np.array([0.8, 0.3, 0, 0.5, 0.05, 0])
    n = 20
    xs = np.sort(np.random.uniform(0, 1, size=n))
    xs[0] = 0
    xs[-1] = 1

    ys = np.random.uniform(0, 1, size=n)
    ys[0] = 0.8
    ys[-1] = 0
    # spl = CubicSpline(xs, ys)
    # akima = Akima1DInterpolator(xs, ys)
    pchip = PchipInterpolator(xs, ys)

    xnew = np.linspace(0, 1, num=1001)

    x_learned = np.linspace(0, 1, num=101)
    # plt.plot(xnew, spl(xnew), '--', label='spline')
    # plt.plot(xnew, akima(xnew), '-', label='Akima')

    plt.plot(xnew, pchip(xnew), '-', label='Pchip')
    plt.plot(x_learned, pchip(x_learned), '-', label='Learned')
    plt.plot(xs, ys, 'o')
    plt.legend()
    plt.savefig('test.png')
    # points = [(0, 0.7), (0.5, 0.4), (0.8, 0.3)]
