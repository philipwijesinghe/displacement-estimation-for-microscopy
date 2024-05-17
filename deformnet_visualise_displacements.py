# -*- coding: utf-8 -*-
""" Plot and calculate error of noise data vs ground truth
"""

import os
from PIL import Image

import numpy as np

import deeplearning.transforms as tf
import matplotlib.pyplot as plt

from helper.visualizer import save_disps_to_phasor, save_vectors

# Run inference code first

# Load inference outputs and evaluate metrics
gt_path = r"C:\example\inference\drosophila"
eval_path = gt_path
suffix = r"inference"

supervised = False

crop = 1024
transforms = [
    tf.NormalizationUniversal(),
    tf.SimultaneousCenterCrop(img_size=[crop, crop]),
    tf.ToNumpy(),
]

# Display
max_mag = 15
max_val = max_mag * 6
scale = 1 / 4
hex_len = 20

bg_remove = 0

# Use mask?
mask_path = os.path.join(gt_path, "mask.png")
if os.path.exists(mask_path):
    mask = Image.open(mask_path)
    mask = np.asarray(mask).astype("uint8")
    mask[mask > 1] = 1
else:
    mask = None


plt.figure()
print(f"{gt_path}")

save_disps_to_phasor(
    os.path.join(gt_path, "Dispx-{0}".format(suffix)),
    os.path.join(gt_path, "Dispy-{0}".format(suffix)),
    os.path.join(gt_path, "Phasor-{0}".format(suffix)),
    max_mag=max_mag,
    background=1,
)

save_vectors(
    os.path.join(gt_path, "Dispx-{0}".format(suffix)),
    os.path.join(gt_path, "Dispy-{0}".format(suffix)),
    os.path.join(gt_path, "Vector-{0}".format(suffix)),
    os.path.join(gt_path, "Ref"),
    scale=scale,
    max_val=max_val,
    hex_len=hex_len,
    mask=mask,
    track=True,
    bg_remove=bg_remove,
)
