# -*- coding: utf-8 -*-
""" Simulate images and displacement data for dl
"""

import os

import numpy
import numpy as np
from PIL import Image

import simulate.imageio as io
from simulate.displacements.cloud_generator import (
    sim_field,
)
from simulate.displacements.cloud_generator import (
    sim_sequence_disp_field as sim_perlin,
)
from simulate.displacements.strainnet_speckle2 import (
    sim_sequence_disp_field as sim_strainnet,
)
from simulate.warper import deform, intensity_modulation


def prepare_strainnet_v2(
    raw_image_folder,
    out_folder,
    n_disps=60,
    img_size=(256, 256),
    noise_strength=0,
    dtype=None,
):
    """Prepares data for dl using simulation code in StrainNet paper

    For each image file in the raw_image_folder, a set of n_disps images are generated, and populate the out_folder
    with Ref, Def, Dispx, Dispy subfolders with .tiff and .npy files
    """
    # Prepare output folders
    out_paths = {
        "Ref": os.path.join(out_folder, "Ref"),
        "Def": os.path.join(out_folder, "Def"),
        "Dispx": os.path.join(out_folder, "Dispx"),
        "Dispy": os.path.join(out_folder, "Dispy"),
    }
    for key, val in out_paths.items():
        if not os.path.exists(val):
            os.makedirs(val)

    img_stack = io.StackReader(raw_image_folder)
    for i, img in enumerate(img_stack):
        img_ref = io.crop(img, img_size, random=False)

        # In StrainNet paper, for one ref image, they generate n=60 number of random displacement fields and save
        disp_x, disp_y = sim_strainnet(subset_size=img_size[0], n_disps=n_disps)

        for n in range(n_disps):
            img_def = deform(img_ref, ux=disp_x[n, :, :], uy=disp_y[n, :, :])

            # add noise to images
            if noise_strength != 0:
                img_ref = add_noise(img_ref, strength=noise_strength)
                img_def = add_noise(img_def, strength=noise_strength)

            img_index = i * n_disps + n
            save_sample(
                out_paths,
                img_ref,
                img_def,
                disp_x[n, :, :],
                disp_y[n, :, :],
                n=img_index,
                dtype=dtype,
            )

            print(f"saving image {img_index} of {len(img_stack) * n_disps}")


def prepare_perlin(
    raw_image_folder,
    out_folder,
    n_disps=1,
    img_size=None,
    noise_strength=0,
    dtype=None,
    modulate_intensity=0,
    random_crop=False,
    **kwargs,
):
    """Prepares data for dl using perlin noise generator"""
    # Prepare output folders
    out_paths = {
        "Ref": os.path.join(out_folder, "Ref"),
        "Def": os.path.join(out_folder, "Def"),
        "Dispx": os.path.join(out_folder, "Dispx"),
        "Dispy": os.path.join(out_folder, "Dispy"),
    }
    if modulate_intensity:
        out_paths["IntMod"] = os.path.join(out_folder, "IntMod")

    for key, val in out_paths.items():
        if not os.path.exists(val):
            os.makedirs(val)

    img_stack = io.StackReader(raw_image_folder)
    for i, img in enumerate(img_stack):
        img_ref = io.crop(img, img_size, random=random_crop) if img_size else img

        disp_x, disp_y = sim_perlin(n_disps=n_disps, shape=img_ref.size, **kwargs)

        for n in range(n_disps):
            img_def = deform(img_ref, ux=disp_x[n, :, :], uy=disp_y[n, :, :])

            if modulate_intensity:
                img_modulate = sim_field(
                    shape=img_ref.size,
                    large_mag=modulate_intensity,
                    perlin_persistence=0.3,
                )
                img_def = intensity_modulation(img_def, img_modulate)
            else:
                img_modulate = 0

            # add noise to images
            if noise_strength != 0:
                img_ref_out = add_noise(img_ref, strength=noise_strength)
                img_def_out = add_noise(img_def, strength=noise_strength)
            else:
                img_ref_out = img_ref
                img_def_out = img_def

            img_index = i * n_disps + n
            save_sample(
                out_paths,
                img_ref_out,
                img_def_out,
                disp_x[n, :, :],
                disp_y[n, :, :],
                n=img_index,
                dtype=dtype,
                out_mod_img=img_modulate,
            )

            print(f"saving image {img_index} of {len(img_stack) * n_disps}")


def save_sample(out_paths, img_ref, img_def, ux, uy, n, dtype=None, out_mod_img=0):
    """Saves an image set to disk

    Based on previous data, types are
    img_def, img_ref are uint8 grayscale tiff
    Dispx and Dispy are numpy array (float64) saves .npy
    """
    out_ref = os.path.join(out_paths["Ref"], "%08i.tiff" % int(n))
    out_def = os.path.join(out_paths["Def"], "%08i.tiff" % int(n))
    out_ux = os.path.join(out_paths["Dispx"], "%08i.npy" % int(n))
    out_uy = os.path.join(out_paths["Dispy"], "%08i.npy" % int(n))
    if type(out_mod_img) is np.ndarray:
        out_mod = os.path.join(out_paths["IntMod"], "%08i.npy" % int(n))
    else:
        out_mod = None

    # TODO: standardise formats - implement in imageio
    if type(img_def) is numpy.ndarray:
        img_def = Image.fromarray(img_def)
        img_def = img_def.convert("L")

    img_ref.save(out_ref)
    img_def.save(out_def)
    if dtype == "float32":
        np.save(
            out_ux, ux.transpose().astype(np.float32)
        )  # Save them in yx coordinates to match visualisation
        np.save(out_uy, uy.transpose().astype(np.float32))
    else:
        np.save(
            out_ux, ux.transpose()
        )  # Save them in yx coordinates to match visualisation
        np.save(out_uy, uy.transpose())
    if type(out_mod_img) is np.ndarray:
        np.save(out_mod, out_mod_img.transpose())


def add_noise(img, strength):
    # add gaussian noise to a PIL image
    img = np.array(img)
    noise = np.ones(img.shape) + np.random.normal(scale=strength, size=img.shape)
    img = np.clip(img * noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img)
