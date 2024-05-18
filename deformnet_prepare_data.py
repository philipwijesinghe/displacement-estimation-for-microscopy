# -*- coding: utf-8 -*-
""" Script for preparing data for deep learning
"""

import os
import shutil

import numpy as np

import simulate.imageio as io
from simulate.simulator import prepare_perlin, prepare_strainnet_v2

""" User Input:

The in_path should be a folder or array of folders that host sequences of images

The out_path will be populated with appropriate test/train/validation data in subfolders
"""
in_path = r"C:\example\drosophila"
out_path = r"C:\example\training\drosophila"

""" Data partitioning

n_disps_per_image: 
    a single ref image is used 'n' times
n_val, n_test: 
    number of validation and test images to separate from raw data (images are exclusive to each set)
randomize:
    shuffle images from raw data
"""
n_disps_per_image = 20
n_val = 10
n_test = 10
randomize = True
img_size = (512, 512)

""" Displacement Method:

options:
    'perlin', 'strainnet'
"""
gen_method = "perlin"

random_crop = True
perlin_persistence = 0.5
final_multiplier = 1  # increase this for higher displacements
modulate_intensity = 0.1 * 255


""" 
MAIN
"""
out_raw = {
    "train": os.path.join(out_path, "train", "Raw"),
    "val": os.path.join(out_path, "val", "Raw"),
    "test": os.path.join(out_path, "test", "Raw"),
}
out_paths = {
    "train": os.path.join(out_path, "train"),
    "val": os.path.join(out_path, "val"),
    "test": os.path.join(out_path, "test"),
}

imgs = io.StackReader(in_path)
print("Total raw images: {0}".format(len(imgs)))

n_train = len(imgs) - n_val - n_test
print("Total images {0}; n_val {1}; n_test {2}".format(len(imgs), n_val, n_test))
if n_train < 1:
    raise ValueError("Not enough images to generate data")

# Populate folders
os.makedirs(out_raw["train"])
if n_val:
    os.makedirs(out_raw["val"])
if n_test:
    os.makedirs(out_raw["test"])

img_idx = np.arange(len(imgs))
if randomize:
    img_idx = np.random.permutation(img_idx)

# Copies images to /raw/ directories (train, val, test)
for i in range(len(imgs)):
    img_path = imgs.files[img_idx[i]]
    out_name = "img_%08i.%s" % (i, img_path[-3:])
    if i < n_val:
        shutil.copy(img_path, os.path.join(out_raw["val"], out_name))
    elif i < n_val + n_test:
        shutil.copy(img_path, os.path.join(out_raw["test"], out_name))
    else:
        shutil.copy(img_path, os.path.join(out_raw["train"], out_name))


def gen_data(in_path_, out_path_):
    if gen_method == "perlin":
        prepare_perlin(
            in_path_,
            out_path_,
            img_size=img_size,
            n_disps=n_disps_per_image,
            perlin_persistence=perlin_persistence,
            final_multiplier=final_multiplier,
            random_crop=random_crop,
            modulate_intensity=modulate_intensity,
        )
    elif gen_method == "strainnet":
        prepare_strainnet_v2(in_path_, out_path_, n_disps=n_disps_per_image)


# Generate data
gen_data(out_raw["train"], out_paths["train"])
if n_val:
    gen_data(out_raw["test"], out_paths["test"])
if n_test:
    gen_data(out_raw["val"], out_paths["val"])
