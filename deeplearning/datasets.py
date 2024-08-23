# -*- coding: utf-8 -*-
""" Pytorch Dataset Classes used by Dataloaders
"""

import glob
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm


class DisplacementDataset(Dataset):
    """Dataset class for loading displacement (Ref)erence; (Def)ormed; (Dispx); (Dispy) data

    data must be in the following folder structure
    dataroot / Ref /
               Def /
               Dispx /
               Dispy /
    """

    def __init__(self, dataroot, transforms_=None, preload=False, suffix=None):
        self.dataroot = dataroot
        self.transform = transforms.Compose(transforms_) if transforms_ else None
        self.preload = preload

        # Assemble file list
        self.files_ref = self._get_file_list("Ref")
        self.files_def = self._get_file_list("Def")

        if suffix:
            self.files_ux = self._get_file_list("Dispx-{0}".format(suffix))
            self.files_uy = self._get_file_list("Dispy-{0}".format(suffix))
        else:
            self.files_ux = self._get_file_list("Dispx")
            self.files_uy = self._get_file_list("Dispy")

        # Preload data?
        self.data_ref = None
        self.data_def = None
        self.data_ux = None
        self.data_uy = None
        if self.preload:
            self._preload_data()

    def _get_file_list(self, subfolder):
        return sorted(glob.glob(os.path.join(self.dataroot, subfolder, "*.*")))

    def _load_single(self, index):
        # noinspection PyTypeChecker
        img_ref = np.asarray(ImageOps.exif_transpose(Image.open(self.files_ref[index])))
        # noinspection PyTypeChecker
        img_def = np.asarray(ImageOps.exif_transpose(Image.open(self.files_def[index])))
        img_ux = np.load(self.files_ux[index])
        img_uy = np.load(self.files_uy[index])
        return img_ref, img_def, img_ux, img_uy

    def _preload_data(self):
        """Preloads all images as np arrays into RAM for speed

        Currently, loads the full FOV which will potentially use a lot of RAM. This is compatible with random
        cropping trainsforms every epoch. RAM efficient option would be to preload cropped FOVs only (i.e.,
        after transforms), but that means crops will not change between epochs.
        """
        img_ref, img_def, img_ux, img_uy = self._load_single(0)
        self.data_ref = np.zeros([len(self), *img_ref.shape], dtype=img_ref.dtype)
        self.data_def = np.zeros([len(self), *img_def.shape], dtype=img_def.dtype)
        self.data_ux = np.zeros([len(self), *img_ux.shape], dtype=img_ux.dtype)
        self.data_uy = np.zeros([len(self), *img_uy.shape], dtype=img_uy.dtype)

        # print(f"Preloading data from {self.dataroot} into RAM")
        for index in tqdm(range(len(self)), leave=True):
            img_ref, img_def, img_ux, img_uy = self._load_single(index)
            self.data_ref[index, :, :] = img_ref
            self.data_def[index, :, :] = img_def
            self.data_ux[index, :, :] = img_ux
            self.data_uy[index, :, :] = img_uy

    def __getitem__(self, index):
        if self.preload:
            img_ref = self.data_ref[index, :, :]
            img_def = self.data_def[index, :, :]
            img_ux = self.data_ux[index, :, :]
            img_uy = self.data_uy[index, :, :]
        else:
            img_ref, img_def, img_ux, img_uy = self._load_single(index)

        sample = {"Ref": img_ref, "Def": img_def, "Dispx": img_ux, "Dispy": img_uy}
        """ We should assert that sample is a dict of numpy arrays """
        if self.transform:
            sample = self.transform(sample)
        """ This must return a torch tensor after transforms """
        return sample

    def __len__(self):
        return len(self.files_ref)


class DisplacementDatasetUnsupervised(Dataset):
    """Dataset class for loading displacement (Ref)erence; (Def)ormed data

    data must be in the following folder structure
    dataroot / Ref /
               Def /
    """

    def __init__(self, dataroot, transforms_=None, preload=False):
        self.dataroot = dataroot
        self.transform = transforms.Compose(transforms_) if transforms_ else None
        self.preload = preload

        # Assemble file list
        self.files_ref = self._get_file_list("Ref")
        self.files_def = self._get_file_list("Def")

        # Preload data?
        self.data_ref = None
        self.data_def = None
        if self.preload:
            self._preload_data()

    def _get_file_list(self, subfolder):
        files_include = glob.glob(os.path.join(self.dataroot, subfolder, "*.*"))
        files_exclude = glob.glob(os.path.join(self.dataroot, subfolder, "*.db"))
        return sorted(set(files_include) - set(files_exclude))

    def _load_single(self, index):
        # noinspection PyTypeChecker
        img_ref = np.asarray(ImageOps.exif_transpose(Image.open(self.files_ref[index])))
        # noinspection PyTypeChecker
        img_def = np.asarray(ImageOps.exif_transpose(Image.open(self.files_def[index])))
        return img_ref, img_def

    def _preload_data(self):
        """Preloads all images as np arrays into RAM for speed

        Currently, loads the full FOV which will potentially use a lot of RAM. This is compatible with random
        cropping trainsforms every epoch. RAM efficient option would be to preload cropped FOVs only (i.e.,
        after transforms), but that means crops will not change between epochs.
        """
        img_ref, img_def = self._load_single(0)
        self.data_ref = np.zeros([len(self), *img_ref.shape], dtype=img_ref.dtype)
        self.data_def = np.zeros([len(self), *img_def.shape], dtype=img_def.dtype)

        for index in range(len(self)):
            img_ref, img_def = self._load_single(index)
            self.data_ref[index, :, :] = img_ref
            self.data_def[index, :, :] = img_def

    def __getitem__(self, index):
        if self.preload:
            img_ref = self.data_ref[index, :, :]
            img_def = self.data_def[index, :, :]
        else:
            img_ref, img_def = self._load_single(index)

        sample = {"Ref": img_ref, "Def": img_def}
        """ We should assert that sample is a dict of numpy arrays """
        if self.transform:
            sample = self.transform(sample)
        """ This must return a torch tensor after transforms """
        return sample

    def __len__(self):
        return len(self.files_ref)


class DisplacementOnlyDataset(Dataset):
    """Dataset class for loading displacement (Dispx); (Dispy) data

    data must be in the following folder structure
    dataroot / Dispx /
               Dispy /
    """

    def __init__(self, dataroot, transforms_=None, preload=False, suffix=None):
        self.dataroot = dataroot
        self.transform = transforms.Compose(transforms_) if transforms_ else None
        self.preload = preload

        # Assemble file list
        if suffix:
            self.files_ux = self._get_file_list("Dispx-{0}".format(suffix))
            self.files_uy = self._get_file_list("Dispy-{0}".format(suffix))
        else:
            self.files_ux = self._get_file_list("Dispx")
            self.files_uy = self._get_file_list("Dispy")

        # Preload data?
        self.data_ux = None
        self.data_uy = None
        if self.preload:
            self._preload_data()

    def _get_file_list(self, subfolder):
        return sorted(glob.glob(os.path.join(self.dataroot, subfolder, "*.*")))

    def _load_single(self, index):
        img_ux = np.load(self.files_ux[index])
        img_uy = np.load(self.files_uy[index])
        return img_ux, img_uy

    def _preload_data(self):
        """Preloads all images as np arrays into RAM for speed

        Currently, loads the full FOV which will potentially use a lot of RAM. This is compatible with random
        cropping trainsforms every epoch. RAM efficient option would be to preload cropped FOVs only (i.e.,
        after transforms), but that means crops will not change between epochs.
        """
        img_ux, img_uy = self._load_single(0)
        self.data_ux = np.zeros([len(self), *img_ux.shape], dtype=img_ux.dtype)
        self.data_uy = np.zeros([len(self), *img_uy.shape], dtype=img_uy.dtype)

        for index in range(len(self)):
            img_ux, img_uy = self._load_single(index)
            self.data_ux[index, :, :] = img_ux
            self.data_uy[index, :, :] = img_uy

    def __getitem__(self, index):
        if self.preload:
            img_ux = self.data_ux[index, :, :]
            img_uy = self.data_uy[index, :, :]
        else:
            img_ux, img_uy = self._load_single(index)

        sample = {"Dispx": img_ux, "Dispy": img_uy}
        """ We should assert that sample is a dict of numpy arrays """
        if self.transform:
            sample = self.transform(sample)
        """ This must return a torch tensor after transforms """
        return sample

    def __len__(self):
        return len(self.files_ref)
