import numpy as np
from numba import njit
from alg.bits import bitstring_to_array, bit_metrics


@njit
def acc_prob_e(score, n_score, T0, k, max_iter, min_score, max_score, a, p_max):
    moment = (k + 1) / max_iter
    x = (n_score - score) / (max_score - min_score) * (1 - moment) + moment
    return p_max * (np.exp((1 - x) * a) - 1) / (np.exp(a) - 1)


@njit
def acc_prob_ln(score, n_score, T0, k, max_iter, min_score, max_score, t, p_max):
    moment = (k + 1) / max_iter
    x = (n_score - score) / (max_score - min_score) * (1 - moment) + moment
    return p_max * np.log(x * (1 - t) + t) / np.log(t)


# scipy + matlab
@njit
def acc_prob_boltz(score, n_score, T0, k, *args):
    T = T0 / np.log(k + 2)
    return np.exp((score - n_score) / T)


# exponential schedule in simulated quenching
@njit
def acc_prob_quench(score, n_score, T0, k, *args):  # exp
    c = 0.9  # must be in (0, 1). Tried 0.99
    T = max(T0 * np.exp((c - 1) * k), 1e-12)
    return np.exp((score - n_score) / T)


# fast or cauchy
@njit
def acc_prob_fast(score, n_score, T0, k, *args):  # fast
    T = T0 / (k + 1)
    return np.exp((score - n_score) / T)


# @njit
def sa(target_function, lb, ub, acc_prob_, T0, apf_param, p_max, n_dims,
       max_iter, tolerance=1e-5, cache_ratio=1, warm_start=False, min_score=0., max_score=0.):
    n_bits_value, solution_powers, max_value_on_bits = bit_metrics(lb, ub, tolerance)

    n_bits = n_dims * n_bits_value
    x = (np.random.uniform(0, 1, n_bits) < 0.5).astype(np.int32)

    xa = bitstring_to_array(x, n_dims, n_bits_value, lb, ub, max_value_on_bits, solution_powers)
    score = target_function(xa)

    scores = np.empty(max_iter + 1)
    accept_probs = np.empty(max_iter + 1)
    outcomes = np.empty(max_iter + 1)

    history = np.empty(max_iter + 1)
    history[0] = score

    if warm_start:
        if score < min_score:
            min_score = score
        if score > max_score:
            max_score = score
    else:
        min_score = score
        max_score = score

    cache_size = int(max_iter * cache_ratio)

    for k in range(max_iter):
        scores[k] = score

        mutation_bit = np.random.randint(0, n_bits)
        neighbor = x.copy()
        neighbor[mutation_bit] = 1 - neighbor[mutation_bit]
        n_score = target_function(bitstring_to_array(neighbor, n_dims, n_bits_value,
                                                     lb, ub, max_value_on_bits, solution_powers))

        history[k + 1] = n_score

        cache = history[max((k + 1) - cache_size, 0):k + 2]
        min_score = min(min_score, np.min(cache))
        max_score = max(max_score, np.max(cache))

        accepted = 0
        if n_score <= score:
            accept_probs[k] = 1
            score = n_score
            x = neighbor
            accepted = 1
        else:
            _p2 = acc_prob_(score, n_score, T0, k, max_iter, min_score, max_score, apf_param, p_max)
            accept_probs[k] = _p2
            if np.random.uniform(0, 1) < _p2:
                score = n_score
                x = neighbor
                accepted = 2
        outcomes[k] = accepted

    # add the final scores
    scores[max_iter] = score
    accept_probs[max_iter] = 1
    outcomes[max_iter] = -1

    return scores, accept_probs, outcomes
