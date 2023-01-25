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


def dirichlet_kernel(order: int = 10, theta: float = np.pi):
    return 1 / 2 + cosine_summation_closed(N=order, theta=theta)


def fejer_kernel(order: int = 10, theta: float = np.pi):
    # assert isinstance(N, int) and N >= 1, "N must be a natural number."
    return (np.sin(order * theta / 2) ** 2 / np.sin(theta / 2) ** 2) / 2 * order


def helper(kernel, filename):
    x = np.linspace(start=1 * -np.pi, stop=1 * np.pi, num=50000)
    for order in range(1, 6):
        plt.plot(
            x,
            kernel(order=order, theta=x),
            label=f"$F_{{{order}}}\\left(\\theta\\right)$",
            lw=0.5,
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename}.pdf")


if __name__ == "__main__":
    helper(fejer_kernel, "plot_fejer")
    # helper(dirichlet_kernel, "plot_dirichlet")
