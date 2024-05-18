# -*- coding: utf-8 -*-
""" Visualizer helper functions
"""

import glob
import os

import numpy as np
import scipy

# from skimage import io
from PIL import Image
from skimage import filters


def load_lut(lutpath):
    import csv

    from matplotlib.colors import ListedColormap

    if not os.path.exists(lutpath):
        print("Cannot find file")
        return 0

    with open(lutpath) as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)
    d = d[1:]
    d = np.array(d).astype("double") / 255  # 0..1

    cmap = ListedColormap(d)
    return cmap


def calc_disp_phasor(disps):
    # disps are [2, H, W] torch tensors
    # TODO: check x,y orientation vs images
    x = disps[0,].squeeze()
    y = disps[1,].squeeze()
    magnitude = np.sqrt(x**2 + y**2)
    angle = np.arctan2(x, y)

    return magnitude, angle


def displacement_phasor(dispx, dispy, max_mag=8):
    mag = (dispx**2 + dispy**2) ** (1 / 2)
    mag = mag / max_mag
    mag[mag > 1] = 1
    ang = np.arctan2(dispx, dispy) / np.pi

    return mag, ang


def strains(dispx, dispy, kernel=5):
    dispx = scipy.ndimage.gaussian_filter(dispx, kernel)
    dispy = scipy.ndimage.gaussian_filter(dispy, kernel)

    dxy, dxx = np.gradient(dispx)
    dyy, dyx = np.gradient(dispy)
    ddiv = dxx + dyy

    # np.linalg.eig
    # dxx = scipy.ndimage.gaussian_filter(dxx, kernel)
    # dyy = scipy.ndimage.gaussian_filter(dyy, kernel)
    # ddiv = scipy.ndimage.gaussian_filter(ddiv, kernel)

    return dxx, dyy, ddiv


def phasor_to_cmap(mag, ang, background=0, cmap=None):
    import matplotlib.pyplot as plt

    ang = (ang + 1) / 2  # [0.0, 1.0]

    if cmap is None:
        cmap = plt.get_cmap("hsv")
    rgb = cmap(ang)
    if background == 0:  # black bg
        rgb = rgb[:, :, :3] * np.repeat(mag[:, :, np.newaxis], 3, axis=2)
    elif background == 1:  # white bg
        complement = 1 - rgb[:, :, :3]
        rgb = rgb[:, :, :3] + complement * (
            1 - np.repeat(mag[:, :, np.newaxis], 3, axis=2)
        )

    return rgb


def save_disps_to_phasor(in_path_ux, in_path_uy, out_path, max_mag=8, background=0):
    files_x = sorted(glob.glob(os.path.join(in_path_ux, "*.npy*")))
    files_y = sorted(glob.glob(os.path.join(in_path_uy, "*.npy*")))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i in range(len(files_x)):
        print("Saving {0} of {1} images".format(i, len(files_x)))
        ux = np.load(files_x[i])
        uy = np.load(files_y[i])
        mag, ang = displacement_phasor(ux, uy, max_mag)
        rgb = phasor_to_cmap(mag, ang, background=background)
        img = Image.fromarray(np.uint8(rgb * 255))
        img.save(os.path.join(out_path, "%08i.tiff" % int(i)))


def save_disps_to_strain(in_path_ux, in_path_uy, out_path, kernel=5):
    files_x = sorted(glob.glob(os.path.join(in_path_ux, "*.npy*")))
    files_y = sorted(glob.glob(os.path.join(in_path_uy, "*.npy*")))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i in range(len(files_x)):
        print("Saving {0} of {1} images".format(i, len(files_x)))
        ux = np.load(files_x[i])
        uy = np.load(files_y[i])

        dx, dyy, ddiv = strains(ux, uy, kernel)
        # rgb = phasor_to_cmap(mag, ang, background=background)
        img = Image.fromarray(ddiv)
        img.save(os.path.join(out_path, "%08i.tiff" % int(i)))


def save_disps_to_tiff(in_path_ux, in_path_uy, out_suffix, max_mag=8):
    files_x = sorted(glob.glob(os.path.join(in_path_ux, "*.npy*")))
    files_y = sorted(glob.glob(os.path.join(in_path_uy, "*.npy*")))
    out_path_ux = in_path_ux + "-" + out_suffix
    out_path_uy = in_path_uy + "-" + out_suffix
    if not os.path.exists(out_path_ux):
        os.makedirs(out_path_ux)
    if not os.path.exists(out_path_uy):
        os.makedirs(out_path_uy)

    for i in range(len(files_x)):
        print("Saving {0} of {1} images".format(i, len(files_x)))
        ux = np.load(files_x[i])
        uy = np.load(files_y[i])
        # ux = ux / max_mag
        # uy = uy / max_mag
        img_ux = Image.fromarray(ux, mode="F")
        img_uy = Image.fromarray(uy, mode="F")
        img_ux.save(os.path.join(out_path_ux, "%08i.tiff" % int(i)))
        img_uy.save(os.path.join(out_path_uy, "%08i.tiff" % int(i)))


def calc_hex_vector_mask(X, Y, hex_len=20):
    # Lets try hex eval
    # bases
    # v1
    x1 = hex_len
    y1 = 0
    # v2 - rotate by 60
    x2 = x1 * 0.5 - y1 * 3 ** (1 / 2) / 2
    y2 = x1 * 3 ** (1 / 2) / 2 + y1 * 0.5
    #
    B = np.array([[x1, x2], [y1, y2]])
    Binv = np.linalg.inv(B)

    a1 = Binv[0, 0] * X + Binv[0, 1] * Y
    a2 = Binv[1, 0] * X + Binv[1, 1] * Y
    a1 = a1 % 1
    a2 = a2 % 1
    dist = (a1**2 + a2**2) ** (1 / 2)

    dist_m = scipy.ndimage.minimum_filter(dist, size=(3, 3))
    mask = dist == dist_m
    return mask


def save_vectors(
    in_path_ux,
    in_path_uy,
    out_path,
    in_path_ref,
    ss=16,
    scale=0.25,
    max_val=0,
    hex_len=0,
    width=0.004,
    mask=None,
    track=False,
    bg_remove=0,
):
    import matplotlib.pyplot as plt

    def crop_center(img, cropx, cropy):
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx]

    files_x = sorted(glob.glob(os.path.join(in_path_ux, "*.npy*")))
    files_y = sorted(glob.glob(os.path.join(in_path_uy, "*.npy*")))
    files_ref = sorted(glob.glob(os.path.join(in_path_ref, "*.tif*"))) + sorted(
        glob.glob(os.path.join(in_path_ref, "*.png*"))
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dpi = 90
    ux = np.load(files_x[0])
    X, Y = np.meshgrid(np.arange(ux.shape[1]), np.arange(ux.shape[0]))
    X = X.astype("double")
    Y = Y.astype("double")
    if hex_len:
        skip = calc_hex_vector_mask(X, Y, hex_len)
    else:
        skip = (slice(None, None, ss), slice(None, None, ss))

    X0 = X.copy()
    Y0 = Y.copy()
    for i in range(len(files_x)):
        print("Saving {0} of {1} images".format(i, len(files_x)))
        ux = np.load(files_x[i])
        uy = np.load(files_y[i])
        img_ref = np.asarray(Image.open(files_ref[i]))

        crop_x = (img_ref.shape[1] - ux.shape[1]) // 2
        crop_y = (img_ref.shape[0] - ux.shape[0]) // 2

        if mask is not None:
            ux *= crop_center(mask, ux.shape[0], ux.shape[1])
            uy *= crop_center(mask, ux.shape[0], ux.shape[1])
        if max_val != 0:
            maxmask = (ux**2 + uy**2) > max_val**2
            ux[maxmask] = 0
            uy[maxmask] = 0

        if bg_remove:
            bg = img_ref
            bg = filters.gaussian(bg, sigma=bg_remove)
            img_ref = img_ref - bg

        fig = plt.figure(figsize=(ux.shape[1] / dpi, ux.shape[0] / dpi), dpi=dpi)
        # plt.quiver(
        #     Y[skip],
        #     X[skip],
        #     ux[skip],
        #     -uy[skip],
        #     width=width,
        #     angles='xy',
        #     scale_units="xy",
        #     scale=scale,
        #     color=(1, 0, 0),
        # )
        plt.quiver(
            X[skip] + crop_x,
            Y[skip] + crop_y,
            ux[skip],
            uy[skip],
            width=width,
            angles="xy",
            scale_units="xy",
            scale=scale,
            color=(1, 0, 0),
        )
        plt.imshow(img_ref, cmap="gray")
        plt.savefig(os.path.join(out_path, "%08i.tiff" % int(i)), dpi=dpi)
        plt.close(fig)

        if track:
            X += ux
            Y += uy
            diff = ((X - X0) ** 2 + (Y - Y0) ** 2) ** (1 / 2)
            diff_mask = diff > hex_len / 2
            X[diff_mask] = X0[diff_mask]
            Y[diff_mask] = Y0[diff_mask]
