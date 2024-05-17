# -*- coding: utf-8 -*-
""" Helper functions to Analyze DL data
"""
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scipy.stats as stats
from scipy.ndimage import gaussian_filter

from deeplearning.multiscaleloss import ssim, lpips_numpy, LPIPS
from simulate.warper import deform


def crop_centre(img, crop=16):
    img = img[crop:-crop, crop:-crop]
    return img


def mse(a, b, map=False, normalize=False, threshold=None):
    """Root mean square error"""
    if threshold is not None:
        a = np.copy(a)
        b = np.copy(b)
        a[np.abs(a) < threshold] = None
        b[np.abs(a) < threshold] = None

    if map:
        return np.abs(a - b)
    else:
        if normalize:
            stdn = np.nanstd(a)
            # a = a
            b = b / np.nanstd(b) * stdn
            diff = a - b
            diff = diff - np.nanmean(diff)
            return np.nanmean(diff**2) ** (1 / 2)
        else:
            return np.nanmean((a - b) ** 2) ** (1 / 2)


def msde(x0, y0, x1, y1, map=False):
    """Root mean squared distance error

    :param x0:
    :param y0:
    :param x1:
    :param y1:
    :return:
    """
    if map:
        return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** (1 / 2)
    else:
        return np.nanmean((x0 - x1) ** 2 + (y0 - y1) ** 2) ** (1 / 2)


def calc_ssim(a, b, map=False):
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        a = to_tensor(a)
        b = to_tensor(b)

    return ssim(a, b, map=map, window_size=11)


def calc_pcc(a, b, map=False, crop=True):
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    if map:
        return (a * b) / np.sqrt(np.nansum(a**2) * np.nansum(b**2))
    else:
        if crop:
            a = crop_centre(a)
            b = crop_centre(b)
        return np.nansum(a * b) / np.sqrt(np.nansum(a**2) * np.nansum(b**2))


def calc_lpips(a, b, lpips_fn, map=False):
    return 1 - lpips_numpy(a, b, lpips_fn)


def to_tensor(img):
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)


def calc_rewarp_pcc(img_ref, img_def, ux, uy, map=False):
    # rewarp - deform deformed image by inverse estimated displacement field
    img_rewarped = deform(img_def, ux, uy)
    # calc pcc
    pcc_val = calc_pcc(img_ref, img_rewarped, map=map)
    return pcc_val, img_rewarped


def calc_pcc_freq(img_ref, img_def, f0=0.5, fw=0.1):
    h, v = img_ref.shape
    vv, hh = np.meshgrid(np.linspace(-1, 1, v), np.linspace(-1, 1, h))
    rr = (vv**2 + hh**2) ** (1 / 2)
    apod = np.exp(-((rr - f0) ** 2) / (2 * fw**2))

    img_ref = np.fft.ifft2(np.fft.fft2(img_ref) * np.fft.ifftshift(apod))
    img_def = np.fft.ifft2(np.fft.fft2(img_def) * np.fft.ifftshift(apod))
    img_ref = np.abs(img_ref)
    img_def = np.abs(img_def)
    pcc_val = calc_pcc(img_ref, img_def)
    return pcc_val, img_ref, img_def


# def calc_psd(img):
#     space_ps = np.abs(np.fft.fftn(img))
#     space_ps *= space_ps
#     space_ac = np.fft.ifftn(space_ps).real.round()
#     space_ac /= space_ac[0, 0, 0]


def calc_rewarp_pcc_forward(img_ref, img_def, ux, uy):
    # rewarp - ux and uy were saved in transpose form because of xy indexing in plot vs save
    img_rewarped = deform(img_ref, ux.transpose(), uy.transpose())
    # # remove dc?
    # img_def = img_def - gaussian_filter(img_def, 10)
    # img_rewarped = img_rewarped - gaussian_filter(img_rewarped, 10)
    # calc pcc
    pcc_val = calc_pcc(img_def, img_rewarped)
    return pcc_val, img_rewarped


def calculate_error(
    dataset_gt, dataset_test, n_images=None, error="mse", normalize=False, filt=0
):
    """Calculates errors vs ground truth

    :param normalize:
    :param dataset_gt: DisplacementDataset Object of ground truth data
    :param dataset_test: DisplacementDataset Object of test data
    :param n_images: (default: None - all) Number of images to compare
    :param error: (default: 'mse') Error metric. Options: 'mse',
    :return: mean, std, errors
    """

    if n_images is None:
        n_images = len(dataset_gt)

    if error == "lpips":
        lpips_fn = LPIPS()

    errors = np.zeros([n_images, 1])
    for i in range(n_images):
        sample_gt = dataset_gt[i]
        sample_test = dataset_test[i]
        dispx_gt = crop_centre(sample_gt["Dispx"])
        dispy_gt = crop_centre(sample_gt["Dispy"])
        dispx_test = crop_centre(sample_test["Dispx"])
        dispy_test = crop_centre(sample_test["Dispy"])
        if normalize:
            dispx_gt = dispx_gt / np.nanstd(dispx_gt) * np.nanstd(dispx_test)
            dispy_gt = dispy_gt / np.nanstd(dispy_gt) * np.nanstd(dispy_test)
            dispx_gt = dispx_gt - np.nanmean(dispx_gt - dispx_test)
            dispy_gt = dispy_gt - np.nanmean(dispy_gt - dispy_test)
        if filt:
            # dispx_gt = gaussian_filter(dispx_gt, sigma=filt)
            # dispy_gt = gaussian_filter(dispy_gt, sigma=filt)
            dispx_test = gaussian_filter(dispx_test, sigma=filt)
            dispy_test = gaussian_filter(dispy_test, sigma=filt)
        if error == "mse":
            errors[i] = (mse(dispx_gt, dispx_test) + mse(dispy_gt, dispy_test)) / 2
        elif error == "msde":  # mean squared distance metric
            errors[i] = msde(dispx_gt, dispy_gt, dispx_test, dispy_test)
        elif error == "ssim":
            errors[i] = (
                calc_ssim(dispx_gt, dispx_test) + calc_ssim(dispy_gt, dispy_test)
            ) / 2
        elif error == "pcc":
            errors[i] = (
                calc_pcc(dispx_gt, dispx_test) + calc_pcc(dispy_gt, dispy_test)
            ) / 2
        elif error == "lpips":
            errors[i] = (
                calc_lpips(dispx_gt, dispx_test, lpips_fn)
                + calc_lpips(dispy_gt, dispy_test, lpips_fn)
            ) / 2

    mean = np.mean(errors)
    std = np.std(errors)

    return mean, std, errors


def calculate_error_psd(
    dataset_gt, dataset_test, n_images=None, error="mse", normalize=False, filt=0
):
    """Calculates errors vs ground truth

    :param normalize:
    :param dataset_gt: DisplacementDataset Object of ground truth data
    :param dataset_test: DisplacementDataset Object of test data
    :param n_images: (default: None - all) Number of images to compare
    :param error: (default: 'mse') Error metric. Options: 'mse',
    :return: mean, std, errors
    """

    if n_images is None:
        n_images = len(dataset_gt)

    sample_gt = dataset_gt[0]
    dispx_gt = crop_centre(sample_gt["Dispx"])
    # errors = np.zeros([n_images, 1])
    psd_mean = np.zeros(
        [
            dispx_gt.shape[0] // 2,
        ]
    )
    for i in range(n_images):
        sample_gt = dataset_gt[i]
        sample_test = dataset_test[i]
        dispx_gt = crop_centre(sample_gt["Dispx"])
        dispy_gt = crop_centre(sample_gt["Dispy"])
        dispx_test = crop_centre(sample_test["Dispx"])
        dispy_test = crop_centre(sample_test["Dispy"])
        if normalize:
            dispx_gt = dispx_gt / np.nanstd(dispx_gt) * np.nanstd(dispx_test)
            dispy_gt = dispy_gt / np.nanstd(dispy_gt) * np.nanstd(dispy_test)
            dispx_gt = dispx_gt - np.nanmean(dispx_gt - dispx_test)
            dispy_gt = dispy_gt - np.nanmean(dispy_gt - dispy_test)
        if filt:
            # dispx_gt = gaussian_filter(dispx_gt, sigma=filt)
            # dispy_gt = gaussian_filter(dispy_gt, sigma=filt)
            dispx_test = gaussian_filter(dispx_test, sigma=filt)
            dispy_test = gaussian_filter(dispy_test, sigma=filt)

        error_x = mse(dispx_gt, dispx_test, map=True)
        error_y = mse(dispy_gt, dispy_test, map=True)
        psd_x, kvals = calc_psd(error_x)
        psd_y, kvals = calc_psd(error_y)
        psd = (psd_x + psd_y) / 2
        psd_mean += psd / n_images

    return psd_mean, kvals


def calc_psd(img):
    npix = img.shape[0]

    fourier_image = np.fft.fftn(img)
    fourier_amplitudes = np.abs(fourier_image) ** 2 / npix ** 2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(
        knrm, fourier_amplitudes, statistic="mean", bins=kbins
    )
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return Abins, kvals


def calculate_error_unsupervised(dataset_test, n_images=None, filt=0):
    """Calculates errors vs ground truth

    :param normalize:
    :param dataset_gt: DisplacementDataset Object of ground truth data
    :param dataset_test: DisplacementDataset Object of test data
    :param n_images: (default: None - all) Number of images to compare
    :param error: (default: 'mse') Error metric. Options: 'mse',
    :return: mean, std, errors
    """

    if n_images is None:
        n_images = len(dataset_test)

    errors = np.zeros([n_images, 1])
    errors_ref = np.zeros([n_images, 1])
    for i in range(n_images):
        sample_test = dataset_test[i]
        ux = crop_centre(sample_test["Dispx"])
        uy = crop_centre(sample_test["Dispy"])
        img_ref = crop_centre(sample_test["Ref"])
        img_def = crop_centre(sample_test["Def"])
        if filt:
            ux = gaussian_filter(ux, sigma=filt)
            uy = gaussian_filter(uy, sigma=filt)

        # smooth
        # gs = 5
        # ux = gaussian_filter(ux, gs)
        # uy = gaussian_filter(uy, gs)

        pcc_ref = calc_pcc(img_ref, img_def)
        # pcc_rewarped, img_rewarp = calc_rewarp_pcc(img_ref, img_def, -ux.transpose(), -uy.transpose())
        # Forward version is more accurate because of finite displacement coordinate system - largrange vs euler
        pcc_rewarped, img_rewarp = calc_rewarp_pcc_forward(img_ref, img_def, ux, uy)

        errors[i] = pcc_rewarped
        errors_ref[i] = pcc_ref

    mean = np.mean(errors)
    std = np.std(errors)

    return mean, std, errors, errors_ref


def calculate_error_map(dispx_gt, dispy_gt, dispx_test, dispy_test, error="mse"):
    """Calculates errors vs ground truth

    :param dispx_gt: Ground truth x displacement tensor
    :param dispy_gt: Ground truth y displacement tensor
    :param dispx_test: Test x displacement tensor
    :param dispy_test: Test y displacement tensor
    :param error: (default: 'mse') Error metric. Options: 'mse', 'msde', 'ssim'
    :return: mean, std, errors
    """

    if error == "mse":
        error_map_x = mse(dispx_gt, dispx_test, map=True, threshold=None)
        error_map_y = mse(dispy_gt, dispy_test, map=True, threshold=None)
    elif error == "msde":  # mean squared distance metric
        error_map_x = msde(dispx_gt, dispy_gt, dispx_test, dispy_test, map=True)
        error_map_y = error_map_x
    elif error == "ssim":
        error_map_x = (
            1 - calc_ssim(dispx_gt, dispx_test, map=True).mean(1)[0, :, :].numpy()
        )
        error_map_y = (
            1 - calc_ssim(dispy_gt, dispy_test, map=True).mean(1)[0, :, :].numpy()
        )
    elif error == "pcc":
        error_map_x = calc_pcc(dispx_gt, dispx_test, map=True)
        error_map_y = calc_pcc(dispy_gt, dispy_test, map=True)
    else:
        raise ValueError("Error metric not recognized")

    return error_map_x, error_map_y


def plot_data(
    dataset_gt,
    dataset_test,
    error=None,
    dataset_ref=None,
    n_image=0,
    return_figure=False,
    normalize=True,
    **kwargs
) -> Optional[Figure]:
    """Retrieves single image data from datasets and plots in a grid. Optionally returns figure object.

    :param dataset_gt: DisplacementDataset Object of ground truth data
    :param dataset_test: DisplacementDataset Object of test data
    :param dataset_ref: (optional) DisplacementDataset Object with reference images
    :param n_image: Image number to display
    :param kwargs: plt.imshow optional keyword arguments
    :return:
    """
    image_dict = {}

    sample = dataset_gt[n_image]
    image_dict["dispx_gt"] = sample["Dispx"]
    image_dict["dispy_gt"] = sample["Dispy"]

    sample = dataset_test[n_image]
    image_dict["dispx_test"] = sample["Dispx"]
    image_dict["dispy_test"] = sample["Dispy"]

    if normalize:
        image_dict["dispx_gt"] = (
            image_dict["dispx_gt"]
            / np.nanstd(image_dict["dispx_gt"])
            * np.nanstd(image_dict["dispx_test"])
        )
        image_dict["dispy_gt"] = (
            image_dict["dispy_gt"]
            / np.nanstd(image_dict["dispy_gt"])
            * np.nanstd(image_dict["dispy_test"])
        )
        image_dict["dispx_gt"] = image_dict["dispx_gt"] - np.nanmean(
            image_dict["dispx_gt"] - image_dict["dispx_test"]
        )
        image_dict["dispy_gt"] = image_dict["dispy_gt"] - np.nanmean(
            image_dict["dispy_gt"] - image_dict["dispy_test"]
        )

    if dataset_ref is not None:
        sample = dataset_ref[n_image]
        image_dict["img_def"] = sample["Def"]
        image_dict["img_ref"] = sample["Ref"]

    if error is not None:
        image_dict["error_map_x"], image_dict["error_map_y"] = calculate_error_map(
            image_dict["dispx_gt"],
            image_dict["dispy_gt"],
            image_dict["dispx_test"],
            image_dict["dispy_test"],
            error=error,
        )

    if return_figure:
        fig = plot_grid(image_dict, return_figure=True, **kwargs)
        return fig
    else:
        plot_grid(image_dict, return_figure=False, **kwargs)


def plot_grid(
    image_dict: dict, return_figure: bool = False, **kwargs
) -> Optional[Figure]:
    """Plots an arbitrary even number of images in two rows.

    :param images: list of images to be plotted
    :param return_figure: if True, the function will return the figure object
    :param kwargs: additional arguments passed to the imshow function
    :return: optionally returns the figure object
    """

    # Check if number of images is even
    images = list(image_dict.values())
    labels = list(image_dict.keys())
    if len(images) % 2 != 0:
        raise ValueError("Number of images must be even")

    # Calculate number of columns based on the number of images
    n_columns = len(images) // 2

    # Create figure and axes
    fig, axs = plt.subplots(2, n_columns)
    fig.set_size_inches(n_columns * 5, 10)  # Assuming each subplot will have width 5

    # Plot images
    for idx, img in enumerate(images):
        row = idx % 2
        col = idx // 2
        if "img" in labels[idx]:
            cmap = plt.get_cmap("gray")
        else:
            cmap = plt.get_cmap("viridis")
        im = axs[row, col].imshow(img, cmap=cmap, **kwargs)
        if "error" in labels[idx]:
            im.set_clim([0, 1])
        axs[row, col].set_title(labels[idx])
        plt.colorbar(im, ax=axs[row, col])

    if return_figure:
        return fig
