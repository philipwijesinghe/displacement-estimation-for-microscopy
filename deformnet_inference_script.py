# -*- coding: utf-8 -*-
""" Run inference using a trained model
"""

import deeplearning.transforms as tf
from deeplearning.inference import Inference

# Data location
model_path = r"C:/example/models/example_model/saved_model/checkpoint.pt"
suffix = r"inference"
test_folder = r"C:/example/inference/drosophila"
crop = 1024
overwrite = False


transforms = [
    tf.NormalizationUnsupervised(),
    tf.SimultaneousCenterCrop(img_size=[crop, crop]),
]


Forward = Inference(model_path)
Forward.paired(
    test_folder,
    test_folder,
    suffix=suffix,
    transforms=transforms,
    overwrite=overwrite,
    save_fmt="both",
)

print("done")
