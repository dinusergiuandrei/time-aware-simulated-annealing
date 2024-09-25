import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator

from alg.bits import bit_metrics, array_to_bitstring, bitstring_to_array
from alg.functions import functions
from alg.sa import sa
import wandb


class GeneticAlgorithm:
    def __init__(self, target_function, n_dims, pop_size, mutation_rate, cx_pool_size, n_points,
                 iter_factor=500, n_bins=1000, tolerance=0.01, use_wandb=False):
        self.target_function = target_function
        self.n_dims = int(n_dims)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.cx_pool_size = cx_pool_size
        self.n_points = n_points
        self.population = None
        self.scores = None

        self.max_iter = iter_factor * (n_dims ** 2)
        self.n_bins = n_bins
        self.tolerance = tolerance

        self.n_prob_bits, self.prob_powers, self.prob_max_on_bits = bit_metrics(0, 1, tolerance)

        # between safe_step and 1 - safe_step there must be at least (self.n_points - 1) * self.tolerance
        # 1 - 2 * (n_points * tolerance) >= n_points * tolerance
        # 1 >= 3 * n_points * tolerance
        # n_points * tolerance <= 1/3
        # ex: tolerance = 0.01 -> n_points <= 100/3
        # assert self.n_points * self.tolerance <= 1 / 3, 'Justification in comments above.'

        self.use_wandb = use_wandb

        if self.use_wandb:
            wandb.init(
                project=f'P2_GA_APF_v2',
                group=f'{self.target_function.name}_d={self.n_dims}_iterfactor={iter_factor}',
                name=f'points={self.n_points}',
                resume='allow',
                reinit=True,
                config={'pop_size': pop_size, 'mutation': mutation_rate, 'cx_pool': cx_pool_size}
            )

    def initialize_population(self):
        self.population = list()

        for _ in range(self.pop_size // 2):
            xs = np.random.uniform(size=self.n_points)
            ys = np.zeros(self.n_points + 2, dtype=np.float64)
            ys[0] = 0.8
            ys[-1] = 0
            t = 1e-4
            ys[1:-1] = np.log(xs * (1 - t) + t) / np.log(t)
            self.population.append((xs, ys))

        while len(self.population) < self.pop_size:
            xs = np.random.uniform(size=self.n_points)
            ys = np.random.uniform(size=self.n_points + 2)
            self.population.append((xs, ys))

    def _prob_array_to_bits(self, a):
        return array_to_bitstring(a.astype(np.float64), self.n_prob_bits, 0, 1, self.prob_max_on_bits)

    def _bits_to_prob_array(self, bits):
        n_points = len(bits) // self.n_prob_bits
        return bitstring_to_array(bits, n_points, self.n_prob_bits, 0, 1,
                                  self.prob_max_on_bits, self.prob_powers)

    def _mutate_array(self, a):
        bits = self._prob_array_to_bits(a)
        mutation_matrix = np.random.uniform(size=bits.shape) < self.mutation_rate
        bits_mutated = np.bitwise_xor(bits, mutation_matrix)

        return self._bits_to_prob_array(bits_mutated)

    def mutate_population(self):
        for ind_index in range(len(self.population)):
            xs, ys = self.population[ind_index]
            new_xs = self._mutate_array(xs)
            new_ys = self._mutate_array(ys)
            self.population[ind_index] = new_xs, new_ys

    def crossover(self):
        new_population = list()
        mating_pool = np.random.choice(np.arange(self.pop_size), self.cx_pool_size, replace=False)
        mating_pool_binary = np.zeros(self.pop_size)
        mating_pool_binary[mating_pool] = 1

        for ind_index in range(self.pop_size):
            new_population.append((self.population[ind_index][0].copy(), self.population[ind_index][1].copy()))

        for p in np.arange(0, self.cx_pool_size, 2):
            parent1 = self.population[mating_pool[p]]
            parent2 = self.population[mating_pool[p + 1]]

            xs1 = self._prob_array_to_bits(parent1[0])
            ys1 = self._prob_array_to_bits(parent1[1])

            xs2 = self._prob_array_to_bits(parent2[0])
            ys2 = self._prob_array_to_bits(parent2[1])

            par_switch_x = (np.random.uniform(0, 1, xs1.shape) < 0.5).astype(np.int8)
            data_x1 = np.where(par_switch_x)[0]

            for i in range(data_x1.shape[0]):
                xs1[data_x1[i]] = xs2[data_x1[i]]
                xs2[data_x1[i]] = xs1[data_x1[i]]

            par_switch_y = (np.random.uniform(0, 1, ys1.shape) < 0.5).astype(np.int8)
            data_y1 = np.where(par_switch_y)[0]
            for i in range(data_y1.shape[0]):
                ys1[data_y1[i]] = ys2[data_y1[i]]
                ys2[data_y1[i]] = ys1[data_y1[i]]

            child1_x = self._bits_to_prob_array(xs1)
            child1_y = self._bits_to_prob_array(ys1)

            child2_x = self._bits_to_prob_array(xs2)
            child2_y = self._bits_to_prob_array(ys2)

            new_population.append((child1_x, child1_y))
            new_population.append((child2_x, child2_y))
        self.population = new_population

    def ind_to_clean_points(self, ind):
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

    def dense_from_ind(self, ind):
        xs, ys = self.ind_to_clean_points(ind)
        pchip = PchipInterpolator(xs, ys)
        xnew = np.linspace(0, 1, num=self.n_bins + 1)
        ynew = pchip(xnew)
        return xnew, ynew

    def evaluate_population(self):
        scores = np.empty(len(self.population), dtype=np.float64)
        for ind_index, ind in enumerate(self.population):
            xnew, ynew = self.dense_from_ind(ind)

            # @njit(cache=True)
            def generated_apf(_score, _n_score, T0, k, max_iter, min_score, max_score, t, p_max):
                moment = (k + 1) / max_iter
                x = (_n_score - _score) / (max_score - min_score) * (1 - moment) + moment
                return ynew[round(x * (len(ynew) - 1))]

            this_ind_scores = []
            for _ in range(30):
                sa_scores, accept_probs, outcomes = sa(target_function=self.target_function.f,
                                                       lb=self.target_function.lb, ub=self.target_function.ub,
                                                       acc_prob_=generated_apf, T0=-1000, apf_param=-1000, p_max=-1000,
                                                       n_dims=self.n_dims, max_iter=self.max_iter)
                this_ind_scores.append(np.min(sa_scores))
            min_score = np.median(this_ind_scores)

            if self.target_function.name == 'Rana':
                real_min = -511.7043 * self.n_dims + 511.68714
                min_score = min_score - real_min
            elif self.target_function.name == 'Eggholder':
                real_min = -915.61991 * self.n_dims + 862.10466
                min_score = min_score - real_min
            scores[ind_index] = min_score
        self.scores = scores

    def select_next_generation(self):
        new_population = list()
        new_scores = np.empty(self.pop_size, dtype=self.scores.dtype)

        scores = -self.scores.copy()

        _scores = np.array(scores)
        _scores -= np.min(_scores)

        ratio = np.sum(_scores) / (self.pop_size - 1)

        si = np.argsort(_scores)

        if ratio == 0:
            diffs = np.ones_like(si)
        else:
            tmp = np.cumsum(_scores[si]) / ratio
            diffs = np.diff(np.floor(tmp)).astype(np.int8)

        new_population.append(self.population[si[-1]])
        new_scores[0] = self.scores[si[-1]]
        last_p = 1

        for p in reversed(range(1, len(si))):
            for _ in range(diffs[p - 1]):
                if last_p < self.pop_size:
                    ind = self.population[si[p]]
                    new_population.append((ind[0].copy(), ind[1].copy()))
                    new_scores[last_p] = self.scores[si[p]].copy()
                    last_p += 1

        while last_p < self.pop_size:
            ind = self.population[si[-1]]
            new_population.append((ind[0].copy(), ind[1].copy()))
            new_scores[last_p] = self.scores[si[-1]]
            last_p += 1

        self.population = new_population
        self.scores = new_scores.copy()

    def get_current_best_ind(self):
        return self.population[np.argmin(self.scores)]

    def store_state(self, folder):
        best_ind = self.get_current_best_ind()
        best_x, best_y = best_ind
        np.save(folder / 'best_x.npy', best_x)
        np.save(folder / 'best_y.npy', best_y)

    def run(self, n_generations, convergence_ratio, experiment_folder):
        self.initialize_population()

        history = []
        generation = 0
        done = False
        last_improvement = -1
        all_time_min = np.inf

        while not done:
            self.mutate_population()
            self.crossover()
            self.evaluate_population()
            self.select_next_generation()
            generation_folder = experiment_folder / f'generation={generation}'
            generation_folder.mkdir(parents=True, exist_ok=True)
            self.store_state(generation_folder)
            generation_dict = {
                'generation': generation,
                'max_score': self.scores.max(),
                'mean_score': self.scores.mean(),
                'min_score': self.scores.min()
            }

            history.append(generation_dict)
            if self.use_wandb:
                best_ind = self.get_current_best_ind()
                xs, ys = self.ind_to_clean_points(best_ind)

                xnew, ynew = self.dense_from_ind(best_ind)

                fig, ax = plt.subplots()
                ax.plot(xnew, ynew, '-', label='Pchip')
                ax.plot(xs, ys, 'o')
                ax.legend()

                # fig.savefig(experiment_folder / f'APF_score={self.scores.min()}.png')

                wandb.log(generation_dict | {'APF': fig, 'AOC%': ynew.mean()})
                plt.close()

            generation += 1

            if self.scores.min() < all_time_min:
                all_time_min = self.scores.min()
                last_improvement = generation

            since_improvement = generation - last_improvement
            if generation > n_generations and since_improvement / generation > convergence_ratio:
                done = True

        history_pd = pd.DataFrame(history)
        history_pd.to_csv(experiment_folder / 'history.csv', index=False)


if __name__ == '__main__':
    pop_size = 32
    mutation_rate = 0.01
    cx_pool_size = 20

    n_dims = 2  # 2 to 10
    n_points = 8  # 1 to 20?
    iter_factor = 50  # 50 500 5000
    target_function = functions[-1]  # 0 to 8

    ga = GeneticAlgorithm(target_function=target_function, n_dims=n_dims, pop_size=pop_size,
                          mutation_rate=mutation_rate, cx_pool_size=cx_pool_size,
                          n_points=n_points, iter_factor=iter_factor, use_wandb=False)
