import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve


def generate_sample(f, x_range, N):
    x = np.linspace(*x_range, N)
    epsilon = np.random.normal(0, 0.5, N)
    y = f(x) + epsilon
    return x, y


def polynomial_regression(x, y, degree):
    A = np.vander(x, degree + 1, increasing=True)
    coeffs = solve(A.T @ A, A.T @ y)
    return coeffs


def polynomial_value(x, coeffs):
    return sum(c * x ** i for i, c in enumerate(coeffs))


orig_function = lambda x: x * np.sin(2 * np.pi * x)

x_train, y_train = generate_sample(orig_function, (-1, 1), 40)
x_test = np.linspace(-1, 1, 100)

degrees = [9]

plt.figure(figsize=(12, 8))
plt.ylim(-2, 2)

colors = ['green', 'blue', 'orange']
plt.scatter(x_train, y_train, color='red', label='Sample')

for deg, color in zip(degrees, colors):
    coeffs = polynomial_regression(x_train, y_train, deg)

    y_res = polynomial_value(x_test, coeffs)

    plt.plot(x_test, y_res, label=f'Polynom with deg = {deg}', color=color)

plt.plot(x_test, orig_function(x_test), label='original func', linestyle='dashed', color="black")
plt.legend()
plt.show()