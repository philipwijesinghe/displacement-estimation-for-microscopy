# -*- coding: utf-8 -*-
""" Implementation of displacement simulation in speckle from StrainNet paper (their version 2)

See: Boukhtache, Seyfeddine, et al. "When deep learning meets digital image correlation." Optics and Lasers in
Engineering 136 (2021): 106308.
"""

import numpy as np
# from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline as Interp  # faster for grid data
from joblib import Parallel, delayed


def sim_sequence_disp_field(subset_size=256, n_disps=60):
    disp_x = np.zeros([n_disps, subset_size, subset_size])
    disp_y = np.zeros([n_disps, subset_size, subset_size])

    for n in range(n_disps):
        if n < 1 / 6 * n_disps:
            level = 2
        elif n < 2 / 6 * n_disps:
            level = 4
        elif n < 3 / 6 * n_disps:
            level = 8
        elif n < 4 / 6 * n_disps:
            level = 16
        elif n < 5 / 6 * n_disps:
            level = 32
        else:
            level = 64

        # pull from random sample if n_disps is not enough to generate all levels
        if n < 6:
            level = 2 ** np.random.randint(1, 7)

        disp_x[n, :, :], disp_y[n, :, :] = sim_single_disp_field(subset_size, level)

    return disp_x, disp_y


def sim_single_disp_field(subset_size=256, level=4):
    xp0 = np.linspace(3, 3 + level, subset_size)
    yp0 = np.linspace(3, 3 + level, subset_size)
    xxp0 = np.linspace(1, level + 3, level + 3)
    yyp0 = np.linspace(1, level + 3, level + 3)

    # random displacements for each point
    f = np.random.randint(-100, 100, [level + 3, level + 3]) / 115
    g = np.random.randint(-100, 100, [level + 3, level + 3]) / 115

    fx = Interp(xxp0, yyp0, f)
    gx = Interp(xxp0, yyp0, g)
    disp_x = fx(xp0, yp0)
    disp_y = gx(xp0, yp0)

    disp_x[0:2, :] = 0
    disp_y[0:2, :] = 0
    disp_x[:, 0:2] = 0
    disp_y[:, 0:2] = 0
    disp_x[-3:-1, :] = 0
    disp_y[-3:-1, :] = 0
    disp_x[:, -3:-1] = 0
    disp_y[:, -3:-1] = 0

    return disp_x, disp_y
