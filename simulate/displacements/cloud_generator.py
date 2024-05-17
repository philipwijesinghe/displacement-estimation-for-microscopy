# -*- coding: utf-8 -*-
""" Displacement simulation based on perlin noise cloud generation
"""

import numpy as np
from perlin_numpy import generate_fractal_noise_2d


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def sim_sequence_disp_field(n_disps=60, **kwargs):
    """ Simulates a sequence of displacement fields
    """
    shape = kwargs["shape"]
    disp_x = np.zeros([n_disps, shape[0], shape[1]])
    disp_y = np.zeros([n_disps, shape[0], shape[1]])

    for n in range(n_disps):
        disp_x[n, :, :], disp_y[n, :, :] = sim_disp_field(**kwargs)

    return disp_x, disp_y


def sim_disp_field(**kwargs):
    """ Simulates a single displacement field
    """
    disp_x = sim_field(**kwargs)
    disp_y = sim_field(**kwargs)

    return disp_x, disp_y


def sim_field(
    shape=(256, 256),
    n_large=6,
    large_loc_std=0.1,
    large_scale=0.3,
    large_scale_std=0.15,
    large_mag=8,
    large_mag_std=0.3,
    perlin_octaves=8,
    perlin_persistence=0.5,
    final_multiplier=1,
):
    """ Simulates a single displacement component
    """
    disp = np.zeros(shape)

    generator = np.random.default_rng()

    # calculate closest possible size (size must be a multiple of lacunarity^(octaves-1)*res)
    allowed_size_x = 2 ** (perlin_octaves - 1) * int(shape[0] / 256)
    allowed_size_y = 2 ** (perlin_octaves - 1) * int(shape[1] / 256)
    noise_shape = [
        int(np.rint(shape[0] / allowed_size_x)) * allowed_size_x,
        int(np.rint(shape[1] / allowed_size_y)) * allowed_size_y,
    ]

    # randomise magnitude, size, and location of large 'clouds'
    mags = generator.normal(
        loc=large_mag, scale=large_mag * large_mag_std, size=n_large
    )
    mags[::2] = mags[::2] * -1  # equal no. of positive & negative displacements

    xsizes = generator.normal(
        loc=shape[0] / 2 * large_scale,
        scale=shape[0] / 2 * large_scale * large_scale_std,
        size=n_large,
    )
    ysizes = generator.normal(
        loc=shape[1] / 2 * large_scale,
        scale=shape[1] / 2 * large_scale * large_scale_std,
        size=n_large,
    )

    ylocs = generator.normal(
        loc=shape[0] / 2, scale=shape[0] / 2 * large_loc_std, size=n_large
    )
    xlocs = generator.normal(
        loc=shape[1] / 2, scale=shape[1] / 2 * large_loc_std, size=n_large
    )

    for i in range(n_large):
        y = gaussian(np.arange(0, shape[0]), ylocs[i], ysizes[i])
        x = gaussian(np.arange(0, shape[1]), xlocs[i], xsizes[i])
        disp += mags[i] * np.multiply(np.repeat(y[:, np.newaxis], shape[1], axis=1), x)

    # multiply by Perlin noise for high spatial frequency texture
    noise = generate_fractal_noise_2d(
        noise_shape,
        (int(shape[0] / 256), int(shape[1] / 256)),
        perlin_octaves,
        perlin_persistence,
    )

    # calculate & perform required crop/pad
    crop = np.array(disp.shape) - np.array(noise.shape)

    noise_pad_x = 0
    noise_pad_y = 0

    if crop[0] > 0:
        noise_pad_x = crop[0]
    elif crop[0] < 0:
        noise_crop_x = -crop[0]
        noise_pad_x = 0
        noise = noise[
            int(np.floor(noise_crop_x / 2)) : -int(np.ceil(noise_crop_x / 2)), :
        ]

    if crop[1] > 0:
        noise_pad_y = crop[1]
    elif crop[1] < 0:
        noise_crop_y = -crop[1]
        noise_pad_y = 0
        noise = noise[
            :, int(np.floor(noise_crop_y / 2)) : -int(np.ceil(noise_crop_y / 2))
        ]

    noise = np.pad(
        noise,
        (
            (int(np.floor(noise_pad_x / 2)), int(np.ceil(noise_pad_x / 2))),
            (int(np.floor(noise_pad_y / 2)), int(np.ceil(noise_pad_y / 2))),
        ),
        mode="constant",
    )

    disp *= noise

    disp *= final_multiplier

    return disp
