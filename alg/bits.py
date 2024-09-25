import numpy as np
from numba import njit


# https://stackoverflow.com/questions/41069825/convert-binary-01-numpy-to-integer-or-binary-string


@njit
def find_int_number_of_bits(lower_bound, upper_bound, tolerance):
    r = upper_bound - lower_bound
    t = int(r / tolerance)
    # c = log2(t)
    c = 0
    while t > 0:
        c += 1
        t = t // 2
    return c


@njit
def bit_metrics(lb, ub, tolerance):
    n_bits = find_int_number_of_bits(lb, ub, tolerance)
    powers = 2 ** np.arange(n_bits)[::-1]
    max_value = (1 << n_bits) - 1
    return n_bits, powers, max_value


@njit  # 464 us -> 136 us
def bits_to_value(bits, lb, ub, max_on_bits, powers):
    return (bits * powers).sum() / max_on_bits * (ub - lb) + lb


@njit
def value_to_bits(x, n_bits, lb, ub, max_on_bits):
    x = round((x - lb) / (ub - lb) * max_on_bits)
    bits = list()
    while x > 0:
        bits.append(x % 2)
        x = x // 2
    while len(bits) < n_bits:
        bits.append(0)
    return np.array(bits, dtype=np.int16)[::-1]


@njit
def value_to_bits_v2(x, n_bits, lb, ub, max_on_bits):  # for time comparison
    x = round((x - lb) / (ub - lb) * max_on_bits)
    bits = np.zeros(n_bits, dtype=np.int16)
    p = n_bits
    while x > 0:
        bits[p] = x % 2
        x = x // 2
        p -= 1

    return bits


@njit  # 535 us -> 209 us
def bitstring_to_array(bits, dims, dim_bits, lb, ub, max_on_bits, powers):
    bits = bits.reshape((dims, dim_bits))  #
    return np.array([bits_to_value(b, lb, ub, max_on_bits, powers) for b in bits])


@njit
def array_to_bitstring(x, dim_bits, lb, ub, max_on_bits):
    r = np.empty((x.shape[0], dim_bits), dtype=np.int16)
    for i, xd in enumerate(x):
        r[i] = value_to_bits(xd, dim_bits, lb, ub, max_on_bits)
    r = r.flatten()
    return r
