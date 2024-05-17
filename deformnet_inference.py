# -*- coding: utf-8 -*-
""" Run inference using a trained model
"""
import argparse

import deeplearning.transforms as tf
from deeplearning.inference import Inference

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="full path to trained model")
parser.add_argument("data", type=str, help="full path to paired data for inference")
parser.add_argument(
    "--suffix",
    type=str,
    default="inference",
    help="suffix to append to inference outputs",
)
parser.add_argument("--crop", type=int, default=0, help="crop data")
parser.add_argument(
    "--overwrite", type=bool, default=False, help="overwrite output if it exists"
)
parser.add_argument(
    "--save_fmt", type=str, default="both", help="save format: 'npy', 'tiff', or 'both'"
)
opt = parser.parse_args()
print(opt)


if opt.crop:
    transforms = [
        tf.NormalizationUnsupervised(),
        tf.SimultaneousCenterCrop(img_size=[opt.crop, opt.crop]),
    ]
else:
    transforms = [
        tf.NormalizationUnsupervised(),
    ]

Forward = Inference(opt.model)
Forward.paired(
    opt.data,
    opt.data,
    suffix=opt.suffix,
    transforms=transforms,
    overwrite=opt.overwrite,
    save_fmt=opt.save_fmt,
)

print("done")
