from math import sqrt


def distance(a, b):
    return sqrt(sum([(a_i - b_i)**2 for a_i, b_i in zip(a, b)]))


def partial_difference_quotient(f, v, i, h):
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=1e-8):
    return [partial_difference_quotient(f, v, i, h) for i, _ in enumerate(v)]


def descent_step(x, direction, step_length):
    return [x_i - step_length * d_i for x_i, d_i in zip(x, direction)]


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
        # TODO: use Wolfe conditions for optimal step length
        x_next = min([descent_step(x, gradient, step)
                      for step in step_lengths], key=f)
        if distance(x, x_next) < eps:
            break
        x = x_next

    print('iterations = ' + str(iterations))
    return x_next


def main():
    print(estimate_gradient(rosenbrock, [1, 1]))    # global minimum
    print(minimize_gradient_descent(rosenbrock, [1.4, 0.5]))
    print(minimize_gradient_descent(matyas, [4.3, 2.92]))
    print(minimize_gradient_descent(beale, [-1.9, -2.3]))


if __name__ == '__main__':
    main()
