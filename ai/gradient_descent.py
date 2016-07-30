from math import sqrt, exp
from simulated_annealing import *
import inspect
from numpy import random


def dot(a, b):
    return sum([a_i * b_i for a_i, b_i in zip(a, b)])


def distance(a, b):
    return sqrt(sum([(a_i - b_i)**2 for a_i, b_i in zip(a, b)]))


def partial_difference_quotient(f, v, i, h):
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=1e-8):
    return [partial_difference_quotient(f, v, i, h) for i, _ in enumerate(v)]


def descent_step(x, direction, step_length):
    return [x_i - step_length * d_i for x_i, d_i in zip(x, direction)]


def step_line_search(f, x, gradient, wolfe_const1=0.1, step_max=10, factor=0.3):
    """Backtracking line search which finds a step length,
    satisfying the first condition of Wolfe for sufficient decrease."""
    step = step_max
    while f(descent_step(x, gradient, step)) > \
            f(x) + wolfe_const1 * step * -dot(gradient, gradient):
        step *= factor
    return step


def rosenbrock(v, a=1, b=100):
    return (a - v[0])**2 + b * (v[1] - v[0]**2)**2


def matyas(v):
    return 0.26 * (v[0]**2 + v[1]**2) - 0.48 * v[0] * v[1]


def beale(v):
    return (1.5 - v[0] + v[0] * v[1])**2 + (2.25 - v[0] + v[0] * (v[1]**2))**2
    + (2.625 - v[0] + v[0] * (v[1]**3))**2


# TODO: domain constraint
def minimize_gradient_descent(f, x0, eps=1e-8):
    step_lengths = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    x = x0
    iterations = 0

    while True:
        iterations += 1
        gradient = estimate_gradient(f, x)
        # x_next = descent_step(x, gradient, 0.01)
        # x_next = descent_step(x, gradient, step_line_search(f, x, gradient))
        x_next = min([descent_step(x, gradient, step)
                      for step in step_lengths], key=f)
        print(x_next, f(x_next))
        if distance(x, x_next) < eps:
            break
        x = x_next

    print('iterations = ' + str(iterations))
    return x_next


def optimal_parameters(nodes, pattern_fn):
    params_count = len(inspect.getargspec(pattern_fn).args) - 1

    def error_fn(v):
        return sum([(pattern_fn(x_i, *v) - y_i)**2 for x_i, y_i in nodes])

    return minimize_gradient_descent(error_fn, [1 for _ in range(params_count)])


def line(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def exponent(x, a, b):
    return a * exp(b * x)


def test_gradient_descent():
    print('gradient descent:')
    print(estimate_gradient(rosenbrock, [1, 1]))    # global minimum
    print(minimize_gradient_descent(rosenbrock, [1.4, 0.5]))
    print(minimize_gradient_descent(matyas, [4.3, 2.92]))
    print(minimize_gradient_descent(beale, [-1.9, -2.3]))


def test_regression():
    print('\nleast square regression:')
    nodes_line = [(1, 3), (4, 10), (6, 20), (8, 30), (9, 34),
                  (11, 40), (13, 43)]
    print(optimal_parameters(nodes_line, line))

    # WIP
    nodes_exp = [(1, 1), (2, 2), (3, 5), (4, 9), (5, 18), (6, 34),
                 (7, 70), (8, 132), (9, 264), (10, 537)]
    print(optimal_parameters(nodes_exp, quadratic))


def neigh(T, x):
    point = np.array(x)
    return random.normal(point, max(T, 0.1), point.shape)


def test_annealer():
    # minimize the Rosenbrock function with Simulated Annealing
    SimulatedAnnealing(True, T_MIN, T_MAX, init,
                       neigh, rosenbrock_energy,
                       accept, cooling).execute()


def main():
    # test_gradient_descent()
    # test_regression()
    test_annealer()


if __name__ == '__main__':
    main()
