import numpy as np
from alg.functions import functions
from alg.mo_utils.selection import sel_nsga_iii, find_ideal_point
from deap import algorithms, base, tools, creator

import array, random, copy


def prepare_toolbox(pop_size, n_generations, problem_instance, selection_func, number_of_variables, bounds_low,
                    bounds_up):
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    toolbox = base.Toolbox()

    toolbox.register('evaluate', problem_instance)
    toolbox.register('select', selection_func)

    toolbox.register("attr_float", uniform, bounds_low, bounds_up, number_of_variables)
    toolbox.register("individual", tools.initIterate, creator.Individual3, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=bounds_low, up=bounds_up, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=bounds_low, up=bounds_up, eta=20.0,
                     indpb=1.0 / number_of_variables)

    toolbox.pop_size = pop_size  # population size
    toolbox.max_gen = n_generations  # max number of iteration
    toolbox.mut_prob = 1 / number_of_variables
    toolbox.cross_prob = 0.3

    return toolbox


def nsga_iii(toolbox, stats=None, verbose=False):
    population = toolbox.population(n=toolbox.pop_size)
    return algorithms.eaMuPlusLambda(population, toolbox,
                                     mu=toolbox.pop_size,
                                     lambda_=toolbox.pop_size,
                                     cxpb=toolbox.cross_prob,
                                     mutpb=toolbox.mut_prob,
                                     ngen=toolbox.max_gen,
                                     stats=stats, verbose=verbose)


interesting_functions = []
for i in range(6):
    original_func_index = i + 3
    if original_func_index == 4:
        original_func_index = 2
    interesting_functions.append(functions[original_func_index])

test_function_index = 5

this_exp_functions = [interesting_functions[i] for i in range(6) if i != test_function_index]


def mo_obj_func(ind):
    ind = np.array(ind)
    return [f(ind) for f in this_exp_functions]


if __name__ == '__main__':
    creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * len(this_exp_functions))
    creator.create("Individual3", array.array, typecode='d',
                   fitness=creator.FitnessMin3)

    number_of_variables = 30
    bounds_low, bounds_up = 0, 1
    pop_size = 100
    n_generations = 200

    toolbox = prepare_toolbox(
        pop_size=pop_size,
        n_generations=n_generations,
        problem_instance=mo_obj_func,
        selection_func=sel_nsga_iii,
        number_of_variables=number_of_variables,
        bounds_low=bounds_low,
        bounds_up=bounds_up)

    pop = toolbox.population(n=pop_size)

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    ideal_point = find_ideal_point(pop)

    stats = tools.Statistics()
    stats.register('pop', copy.deepcopy)

    res, logbook = nsga_iii(toolbox, stats=stats)

    pops = logbook.select('pop')

    print(pop)
