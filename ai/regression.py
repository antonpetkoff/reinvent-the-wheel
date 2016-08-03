import inspect
from math import exp
from gradient_descent import *
from simulated_annealing import *


def optimal_parameters(nodes, pattern_fn):
    params_count = len(inspect.getargspec(pattern_fn).args) - 1

    def error_fn(v):
        return sum([(pattern_fn(x_i, *v) - y_i)**2 for x_i, y_i in nodes])

    # return minimize_gradient_descent(error_fn, [1 for _ in range(params_count)])
    return SimulatedAnnealing(error_fn, [1 for _ in range(params_count)]).execute()


def line(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def exponent(x, a, b):
    return a * exp(b * x)
