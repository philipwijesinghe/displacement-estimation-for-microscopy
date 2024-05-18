# -*- coding: utf-8 -*-
""" Functions that warp reference images based on displacement field
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


def deform(img_ref, ux, uy):
    # we assume that u is in pixel coordinates
    img_ref = np.asarray(img_ref)
    assert img_ref.shape == ux.shape

    img_size = img_ref.shape
    x = np.arange(img_size[0])
    y = np.arange(img_size[1])
    y0, x0 = np.meshgrid(y, x)  # because meshgrid is backwards ordering

    xp = x0.flatten() - ux.flatten()
    yp = y0.flatten() - uy.flatten()

    f = RectBivariateSpline(y, x, img_ref)
    img_def = f(yp, xp, grid=False)
    img_def = img_def.reshape(img_size).transpose()

    # Below should be done at the final stage
    # img_def = Image.fromarray(img_def)
    # img_def = img_def.convert("L")

    return img_def


def intensity_modulation(img_ref, img_modulation):
    # img_modulation should be in the [0, 255] dynamic range

    img_def = img_ref + img_modulation

    img_def = np.clip(img_def, 0, 255).astype(np.uint8)

    return img_def
