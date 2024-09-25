'''
n_dims is fixed = 2
iter_factor = 500
'''
from pathlib import Path
import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator
from alg.bits import bit_metrics, array_to_bitstring, bitstring_to_array
from alg.functions import functions
from alg.mo_utils.ngsa3_utils import prepare_toolbox, nsga_iii
from alg.mo_utils.selection import sel_nsga_iii, find_ideal_point
from alg.sa import sa
from deap import algorithms, base, tools, creator
import array, random, copy
from time import time

interesting_functions = []
for i in range(6):
    original_func_index = i + 3
    if original_func_index == 4:
        original_func_index = 2
    interesting_functions.append(functions[original_func_index])


def ind_to_clean_points(ind):
    xs, ys = ind
    xs = np.array([0] + list(xs) + [1])

    xs_to_ys = dict()
    for x, y in zip(xs, ys):
        if x in xs_to_ys.keys():
            xs_to_ys[x].append(y)
        else:
            xs_to_ys[x] = [y]

    new_x = []
    new_y = []
    for x, ys in xs_to_ys.items():
        new_x.append(x)
        new_y.append(np.mean(ys))

    xs = np.array(new_x)
    ys = np.array(new_y)

    ys = ys[np.argsort(xs)]
    xs = xs[np.argsort(xs)]
    return xs, ys


def dense_from_ind(ind, n_bins):
    xs, ys = ind_to_clean_points(ind)
    pchip = PchipInterpolator(xs, ys)
    xnew = np.linspace(0, 1, num=n_bins + 1)
    ynew = pchip(xnew)
    return xnew, ynew


class NGSA3:
    def __init__(self, n_generations, n_sa_repetitions, pop_size, n_points,
                 n_dims=2, iter_factor=500, n_bins=1000, tolerance=0.01):
        self.worker_index = self.get_worker_index()

        self.test_function_index = self.worker_index - 1
        self.n_dims = int(n_dims)
        self.n_sa_repetitions = n_sa_repetitions
        self.pop_size = pop_size
        self.n_points = n_points

        self.max_iter = iter_factor * (n_dims ** 2)
        self.n_bins = n_bins
        self.tolerance = tolerance

        self.n_prob_bits, self.prob_powers, self.prob_max_on_bits = bit_metrics(0, 1, tolerance)

        self.number_of_variables = 2 * n_points + 2

        self.this_exp_functions = [interesting_functions[i]
                                   for i in range(6) if i != self.test_function_index]

        creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * len(self.this_exp_functions))
        creator.create("Individual3", array.array, typecode='d', fitness=creator.FitnessMin3)

        self.toolbox = prepare_toolbox(pop_size=self.pop_size, n_generations=n_generations,
                                       problem_instance=self.generate_mo_obj_func(),
                                       selection_func=sel_nsga_iii,
                                       number_of_variables=self.number_of_variables,
                                       bounds_low=0, bounds_up=1)

        self.pop = self.toolbox.population(n=self.pop_size)

        for ind in self.pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        self.stats = tools.Statistics()
        self.stats.register('pop', copy.deepcopy)

        self.experiment_folder = self.get_experiments_root_folder() / self.get_experiment_name()
        self.experiment_folder.mkdir(parents=True, exist_ok=True)

    def get_worker_index(self):
        if 'win' in sys.platform:
            return 1
        else:
            import socket
            ip = [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close())
                  for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
            return int(ip[-1])

    def get_experiments_root_folder(self):
        if 'win' in sys.platform:
            return Path('H:\\Data') / 'nextcloud2' / f'aiworker{self.worker_index}' / 'experiments'
        else:
            nc_root = Path('~/Nextcloud').expanduser()
            return nc_root / f'aiworker{self.worker_index}' / 'experiments'

    def get_experiment_name(self):
        return f'NGSAIII_tf={self.test_function_index}_pts={self.n_points}'

    def evaluate_ind(self, ind):
        ind = np.array(ind)
        x = ind[:self.n_points]
        y = ind[self.n_points:]
        xnew, ynew = dense_from_ind((x, y), n_bins=self.n_bins)

        def generated_apf(_score, _n_score, T0, k, max_iter, min_score, max_score, t, p_max):
            moment = (k + 1) / max_iter
            x = (_n_score - _score) / (max_score - min_score) * (1 - moment) + moment
            return ynew[round(x * (len(ynew) - 1))]

        this_ind_scores = []

        for target_function in self.this_exp_functions:
            this_f_scores = []
            for _ in range(self.n_sa_repetitions):
                sa_scores, accept_probs, outcomes = sa(target_function=target_function.f,
                                                       lb=target_function.lb, ub=target_function.ub,
                                                       acc_prob_=generated_apf, T0=-1000, apf_param=-1000, p_max=-1000,
                                                       n_dims=self.n_dims, max_iter=self.max_iter)
                this_f_scores.append(np.min(sa_scores))
            min_score = np.median(this_f_scores)

            if target_function.name == 'Rana':
                real_min = -511.7043 * self.n_dims + 511.68714
                min_score = min_score - real_min
            elif target_function.name == 'Eggholder':
                real_min = -915.61991 * self.n_dims + 862.10466
                min_score = min_score - real_min
            this_ind_scores.append(min_score)

        return this_ind_scores

    def generate_mo_obj_func(self):
        return lambda ind: self.evaluate_ind(ind)

    def store_state(self, last_pop):
        xs = []
        ys = []
        for ind in last_pop:
            xs.append(np.array(ind))
            ys.append(np.array(ind.fitness.values))
        xs = np.array(xs)
        ys = np.array(ys)
        np.save(self.experiment_folder / 'last_x.npy', xs)
        np.save(self.experiment_folder / 'last_y.npy', ys)

    def run(self, verbose=False):
        last_pop, logbook = nsga_iii(self.toolbox, stats=self.stats, verbose=verbose)
        self.store_state(logbook.select('pop')[-1])  # should be equal to last_pop


# 1 sa -> 128s
# 5 sa -> 256s
# 30 sa -> 1116s (8 times more)
# when running the master, do it from the remote server
if __name__ == '__main__':
    n_sa_repetitions = 5
    n_generations = 5
    # from 1 to 20
    n_points = 20
    for n_points in [2, 5, 10, 20]:
    # for n_sa_repetitions in [5]:  # [10, 20, 30]
        experiment = NGSA3(n_generations=n_generations, n_sa_repetitions=n_sa_repetitions,
                           pop_size=100, n_points=n_points)

        start_time = time()
        experiment.run(verbose=True)
        total_time = time() - start_time

        s_per_generation = total_time / (n_generations + 1)
        gen_per_s = 1 / s_per_generation
        gen_per_h = gen_per_s * 3600
        print(f'Run time: {total_time}. Suggested for 24h: {24 * gen_per_h}. Expected h 1000 iter: {1000 / gen_per_h}')
