import inspect
from math import exp
from gradient_descent import *
from simulated_annealing import *
from utils import *


def optimal_parameters(nodes, pattern_fn, method):
    params_count = len(inspect.getargspec(pattern_fn).args) - 1

    def error_fn(v):
        return sum([(pattern_fn(x_i, *v) - y_i)**2 for x_i, y_i in nodes])

    def plot_fn(*v):
        return error_fn(v)
    plot(plot_fn, 1, (-100, 100), (-100, 100))

    return method(error_fn, [1 for _ in range(params_count)])


def line(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def exponent(x, a, b):
    return a * exp(b * x)
