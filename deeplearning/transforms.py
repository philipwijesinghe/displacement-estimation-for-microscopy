# -*- coding: utf-8 -*-
""" Transforms used by dataset/dataloader
"""

import torch
import torchvision.transforms.functional as tf

import numpy as np

from torchvision.transforms import transforms


class AddGaussNoise(torch.nn.Module):
    def __init__(self, noise_std_pc=0.05):
        super().__init__()
        self.noise = noise_std_pc
        # torch in is norm to [0, 1] float

    def forward(self, sample):
        if self.noise:
            sample["Ref"] = (
                sample["Ref"] + torch.randn(sample["Ref"].size()) * self.noise
            )
            sample["Def"] = (
                sample["Def"] + torch.randn(sample["Def"].size()) * self.noise
            )

        return sample


class AddPoissonNoise(torch.nn.Module):
    def __init__(self, noise_std_pc=0.05):
        """ Adds Possion noise to images

        I = I * (1 + N(mu=0, std)), where N is a Gaussian noise process

        @param noise_std_pc: standard deviation of noise
        """
        super().__init__()
        self.noise = noise_std_pc
        # torch in is norm to [0, 1] float

    def forward(self, sample):
        if self.noise:
            sample["Ref"] = sample["Ref"] * (
                1 + torch.randn(sample["Ref"].size()) * self.noise
            )
            sample["Def"] = sample["Def"] * (
                1 + torch.randn(sample["Def"].size()) * self.noise
            )

        return sample


class Normalization(torch.nn.Module):
    def __init__(self, mean=0, std=255, mean_disp=0, std_disp=1):
        """ Normalizes and converts displacement data to tensor

        The outputs should fit within a [0, 1] float range; mean and std values must be chosen accordingly

        :param mean: image mean shift
        :param std: image standard deviation
        :param mean_disp: displacement mean shift
        :param std_disp: displacement standard deviation
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.mean_disp = mean_disp
        self.std_disp = std_disp

    def forward(self, sample):
        img_ref, img_def, img_ux, img_uy = (
            sample["Ref"],
            sample["Def"],
            sample["Dispx"],
            sample["Dispy"],
        )

        return {
            "Ref": torch.from_numpy((img_ref - self.mean) / self.std)
            .float()
            .unsqueeze(0),
            "Def": torch.from_numpy((img_def - self.mean) / self.std)
            .float()
            .unsqueeze(0),
            "Dispx": torch.from_numpy((img_ux - self.mean_disp) / self.std_disp)
            .float()
            .unsqueeze(0),
            "Dispy": torch.from_numpy((img_uy - self.mean_disp) / self.std_disp)
            .float()
            .unsqueeze(0),
        }


class NormalizationUniversal(torch.nn.Module):
    def __init__(self, mean=0, std=255, mean_disp=0, std_disp=1):
        """ Normalizes and converts displacement data to tensor

        Testing application to all subsets of availabel data
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.mean_disp = mean_disp
        self.std_disp = std_disp

    def forward(self, sample):
        out = {}
        if "Ref" in sample:
            img_ref = sample["Ref"]
            out["Ref"] = (
                torch.from_numpy((img_ref - self.mean) / self.std).float().unsqueeze(0)
            )
        if "Def" in sample:
            img_def = sample["Def"]
            out["Def"] = (
                torch.from_numpy((img_def - self.mean) / self.std).float().unsqueeze(0)
            )
        if "Dispx" in sample:
            img_ux = sample["Dispx"]
            out["Dispx"] = (
                torch.from_numpy((img_ux - self.mean_disp) / self.std_disp)
                .float()
                .unsqueeze(0)
            )
        if "Dispy" in sample:
            img_uy = sample["Dispy"]
            out["Dispy"] = (
                torch.from_numpy((img_uy - self.mean_disp) / self.std_disp)
                .float()
                .unsqueeze(0)
            )

        # img_ref, img_def, img_ux, img_uy = sample['Ref'], sample['Def'], sample['Dispx'], sample['Dispy']

        return out


class NormalizationUnsupervised(torch.nn.Module):
    def __init__(self, mean=0, std=255):
        """ Normalizes and converts data to tensor

        The outputs should fit within a [0, 1] float range; mean and std values must be chosen accordingly

        :param mean: image mean shift
        :param std: image standard deviation
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        img_ref, img_def = sample["Ref"], sample["Def"]

        self.mean = 0
        self.std = 255

        return {
            "Ref": torch.from_numpy((img_ref - self.mean) / self.std)
            .float()
            .unsqueeze(0),
            "Def": torch.from_numpy((img_def - self.mean) / self.std)
            .float()
            .unsqueeze(0),
        }


class SimultaneousRotate(torch.nn.Module):
    def __init__(self, degrees=(0, 360)):
        super().__init__()
        self.degrees = degrees

    def forward(self, sample):
        angle = transforms.RandomRotation.get_params(degrees=self.degrees)

        # Rotate all images in the set
        for key, img in sample.items():
            sample[key] = tf.rotate(img, angle=angle)

        # Rotate displacement vectors via rotation matrix
        ux = sample["Dispx"]
        uy = sample["Dispy"]

        rot = np.deg2rad(angle)
        sample["Dispx"] = np.cos(rot) * ux - np.sin(rot) * uy
        sample["Dispy"] = np.sin(rot) * ux + np.cos(rot) * uy

        return sample


class SimultaneousDiscreteRotate(torch.nn.Module):
    def __init__(self):
        """ Discrete random rotations in [0, 90, 180, 270]
        """
        super().__init__()

    def forward(self, sample, angle=None):
        if angle is None:
            angle = torch.randint(0, 3, (1,)).item() * 90

        for key, img in sample.items():
            sample[key] = tf.rotate(img, angle=angle, expand=True)

        # Rotate displacement vectors via rotation matrix
        # The x,y coordinates are actually transposed because of visualisation
        ux = sample["Dispy"]
        uy = sample["Dispx"]

        rot = np.deg2rad(angle)
        sample["Dispy"] = np.cos(rot) * ux - np.sin(rot) * uy
        sample["Dispx"] = np.sin(rot) * ux + np.cos(rot) * uy

        return sample


class SimultaneousRotate90(torch.nn.Module):
    def __init__(self):
        """ Discrete random rotations in [0, 90, 180, 270]
        """
        super().__init__()

    def forward(self, sample):
        rotate = torch.rand(1) > 0.5

        if rotate:
            for key, img in sample.items():
                sample[key] = torch.rot90(img, 1, (1, 2))

            # Rotate displacement vectors via rotation matrix
            # The x,y coordinates are actually transposed because of visualisation
            ux = sample["Dispx"]
            uy = sample["Dispy"]
            sample["Dispx"] = uy
            sample["Dispy"] = -ux

        return sample


class SimultaneousFlips(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        flip_h = torch.rand(1) > 0.5
        flip_v = torch.rand(1) > 0.5

        if flip_h:
            for key, img in sample.items():
                sample[key] = img.flip(1)
            sample["Dispx"] = -sample["Dispx"]

        if flip_v:
            for key, img in sample.items():
                sample[key] = img.flip(2)
            sample["Dispy"] = -sample["Dispy"]

        return sample


class SimultaneousCrop(torch.nn.Module):
    def __init__(self, img_size=(256, 256), params=None):
        super().__init__()
        self.img_size = img_size
        self.params = params

    def forward(self, sample):
        if self.params is None:
            [p0, p1, p2, p3] = transforms.RandomCrop.get_params(
                img=sample[next(iter(sample))], output_size=self.img_size
            )
        else:
            [p0, p1, p2, p3] = self.params

        for key, value in sample.items():
            sample[key] = tf.crop(value, p0, p1, p2, p3)

        return sample


class SimultaneousCenterCrop(torch.nn.Module):
    def __init__(self, img_size=(256, 256)):
        super().__init__()
        self.img_size = img_size

    def forward(self, sample):
        crop = transforms.CenterCrop(self.img_size)

        for key, value in sample.items():
            sample[key] = crop(value)

        return sample


# # Transpose is flip + rotate
# class SimultaneousTranspose(torch.nn.Module):
#     def __init__(self):
#         # TODO: This needs displacements to be reassigned with appropriate directions
#         raise ValueError('This transform is not working')
#         super().__init__()
#
#     def forward(self, sample):
#         permute = torch.rand(1) > 0.5
#
#         for key, img in sample.items():
#             if permute:
#                 sample[key] = img.permute([0, 2, 1])
#
#         return sample


# Augment isn't used
class SimultaneousAugment(torch.nn.Module):
    def __init__(self, img_size=(256, 256)):
        super().__init__()
        self.img_size = img_size

    def forward(self, sample):
        [p0, p1, p2, p3] = transforms.RandomCrop.get_params(
            img=sample["Ref"], output_size=self.img_size
        )
        [p4, p5, p6, p7] = transforms.RandomAffine.get_params(
            img_size=self.img_size,
            degrees=[0, 180],
            translate=None,
            scale_ranges=None,
            shears=[0, 10, 0, 10],
        )
        """ 
        Not sure why there is a shear being applied?
        Also, this seems to lead to lots of edge cases 
        """

        for key, value in sample.items():
            value = tf.affine(value, p4, p5, p6, p7)
            sample[key] = tf.crop(value, p0, p1, p2, p3)

        return sample


class ToNumpy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        for key, value in sample.items():
            value = value.detach().cpu().numpy().squeeze()
            sample[key] = value

        return sample
