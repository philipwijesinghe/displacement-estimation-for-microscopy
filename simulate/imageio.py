# -*- coding: utf-8 -*-
""" Image IO functions
"""

import os
import glob

# from skimage import io
from PIL import Image
from torchvision.transforms import transforms


def read(img_path):
    img = Image.open(img_path)
    # TODO: format standardisation
    return img


def write(img_path, img):
    img.save(img_path)
    # TODO: standardise save format
    pass


def crop(img, crop_size, random=True):
    """ Crops PIL or Tensor image to size
    """
    if random:
        tf = transforms.RandomCrop(size=crop_size)
    else:
        tf = transforms.CenterCrop(size=crop_size)
    return tf(img)


class StackReader:
    def __init__(self, folder_path):
        if type(folder_path) is list:
            self.files = []
            for folder_ in folder_path:
                self.files += glob.glob(os.path.join(folder_, "*.png"))
                self.files += glob.glob(os.path.join(folder_, "*.tif"))
                self.files += glob.glob(os.path.join(folder_, "*.tiff"))
        else:
            self.files = glob.glob(os.path.join(folder_path, "*.png"))
            self.files += glob.glob(os.path.join(folder_path, "*.tif"))
            self.files += glob.glob(os.path.join(folder_path, "*.tiff"))

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            img = self._read_img(self.n)
            self.n += 1
            return img
        else:
            raise StopIteration

    def __call__(self, n):
        return self._read_img(n)

    def _read_img(self, n):
        img = read(self.files[n])
        return img
