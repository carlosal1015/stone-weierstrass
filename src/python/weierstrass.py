#!/usr/bin/env python

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from typing import Callable
import matplotlib.pyplot as plt


def weierstrass_transformation(f: Callable, x: float = 0) -> float:
    kernel: Callable = lambda y: f(y) * np.exp(-((x - y) ** 2) / 4)
    result, _ = quad(kernel, -np.Infinity, np.Infinity)
    return result / np.sqrt(4 * np.pi)


# weierstrass_transformation_vectorized = np.fromfunction(
#     function=np.vectorize(weierstrass_transformation[0]), dtype=np.float64
# )

# x = np.linspace(0, 1)

# print(weierstrass_transformation_vectorized(x))

fun2int = lambda x, y: np.exp(-((x - y) ** 2) / 4) / np.sqrt(4 * np.pi)
intfun = lambda x: quad(fun2int, -np.Infinity, np.Infinity, args=(x))[0]
vec_int = np.vectorize(intfun)
x = np.linspace(-10, 20, 5000)
plt.plot(x, vec_int(x))
plt.show()
