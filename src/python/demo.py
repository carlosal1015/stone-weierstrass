#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

dpi, fontsize = 120, 8
plt.rc("font", size=fontsize)
plt.rc("figure", dpi=dpi)
plt.rc("text", usetex=True)
plt.rc("grid", linestyle="dotted")


def cosine_summation_closed(N: int = 10, theta: float = np.pi):
    assert isinstance(N, int) and N >= 1, "N must be a natural number."
    return (np.sin(2 * (N + 1) * theta / 2)) / (2 * np.sin(theta / 2)) - 1 / 2


def cosine_summation(N: int = 10, theta: float = np.pi):
    assert isinstance(N, int) and N >= 1, "N must be a natural number."
    k = np.arange(1, N + 1)
    return np.sum(np.cos(k * theta))


# def predicted_cosine_summation()
# def sine_summation(N: int = 10, theta: float = np.pi):
#     assert isinstance(N, int) and N >= 1, "N must be a natural number."
#     return np.sin(N * theta) ** 2 / np.sin(theta)


def dirichlet_kernel(order: int = 10, theta: float = np.pi):
    return 1 / 2 + cosine_summation_closed(N=order, theta=theta)


def fejer_kernel(order: int = 10, theta: float = np.pi):
    return (np.sin(order * theta / 2) ** 2 / np.sin(theta / 2) ** 2) / 2 * order


# lower: int = 1, upper: int = 10
# range(lower, upper + 1, 2)
def helper(kernel, filename):
    x = np.linspace(start=-np.pi, stop=np.pi, num=5000)
    for order in range(1, 6):
        plt.plot(
            x,
            kernel(order=order, theta=x),
            label=f"$F_{{{order}}}\\left(\\theta\\right)$",
            lw=0.5,
        )
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename}.pdf")


# helper(dirichlet_kernel, "plot_dirichlet")
helper(fejer_kernel, "plot_fejer")
