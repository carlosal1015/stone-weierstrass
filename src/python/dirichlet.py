#!/usr/bin/env python

import numpy as np


def sum_cosine(n: int, theta: float):
    return np.sum(np.cos(theta))


# def dirichlet(x, N):
#     return np.piecewise(
#         x, [x == 0, x != 0], [N, lambda x: np.sin(N * np.pi * x) / np.sin(np.pi * x)]
#     )

#     # x = [val * radians for val in np.arange(0, 15, 0.01)]
#     # plt.plot(x, cos(x), xunits=radians)
#     plt.tight_layout()
