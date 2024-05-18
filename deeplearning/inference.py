# -*- coding: utf-8 -*-
""" Inference using trained strainnet type model

We need two methods for inference - one is paired Ref/ Def/ images and one for sequential images
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import deeplearning.config as config_handler
import deeplearning.models as models
import deeplearning.transforms as tf
from deeplearning.datasets import DisplacementDatasetUnsupervised


class Inference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        network_data = torch.load(model_path, map_location=self.device)

        self.conf = config_handler.load(
            os.path.abspath(os.path.join(model_path, "../../", "config_run.yml"))
        )

        # TODO: This is likely incorrect, standardize torch.save .load methods
        self.model = models.__dict__[self.conf["model"]]()
        self.model.load_state_dict(network_data["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def paired(
        self, datapath, outpath, suffix="inference", transforms=None, overwrite=True, save_fmt='npy'
    ):
        """ Runs inference on paired image data in /Ref/ and /Def/ folders

        :param datapath: path to folder containing /Ref/ and /Def/
        :param outpath: path to output folder
        :param suffix: append inference output with desired suffix, e.g., Dispx-<suffix>
        :param transforms: tf.transforms for preprocessing
        :param overwrite: overwrite existing data if exists
        :param save_fmt: ["npy", "tiff", "both"] output format
        """

        """Load and process paired images in datapath/Ref and ./Def"""

        save_paths = {
            "root": datapath,
            "out_ux": os.path.join(outpath, f"Dispx-{suffix}"),
            "out_uy": os.path.join(outpath, f"Dispy-{suffix}"),
        }
        if os.path.exists(save_paths["out_ux"]) and not overwrite:
            print(
                f"Output path exists. Code terminated to not overwrite data. Path: {save_paths['out_ux']}"
            )
            return

        for key, val in save_paths.items():
            if not os.path.exists(val):
                os.makedirs(val)

        if transforms is not None:
            transforms_ = transforms
        else:
            transforms_ = [tf.NormalizationUnsupervised()]

        dataset = DisplacementDatasetUnsupervised(
            save_paths["root"], transforms_=transforms_, preload=False
        )
        dataloader = DataLoader(dataset=dataset, batch_size=1)

        norm_params = self.conf["transforms"]["normalisation"]
        mean_disp = 0
        std_disp = 1
        if "mean_disp" in norm_params:
            mean_disp = norm_params["mean_disp"]
        if "std_disp" in norm_params:
            std_disp = norm_params["std_disp"]

        for i, batch in enumerate(dataloader):
            print("Inference {0} of {1}".format(i, len(dataloader)))

            inputs = self._load_batch_unsupervised(batch)
            with torch.no_grad():
                output = self.model(inputs)

            disp_output = output.detach().cpu().squeeze().numpy()
            disp_output = (disp_output / std_disp) + mean_disp
            # TODO: displacement normalisation at training level should be saved and rolled back here - needs studies
            #  of abs value diffs
            ux = disp_output[0, :, :]
            uy = disp_output[1, :, :]

            if save_fmt in ["npy", "both"]:
                save_ux = os.path.join(save_paths["out_ux"], "%08i.npy" % int(i))
                save_uy = os.path.join(save_paths["out_uy"], "%08i.npy" % int(i))
                np.save(save_ux, ux)
                np.save(save_uy, uy)
            if save_fmt in ["tiff", "both"]:
                save_ux = os.path.join(save_paths["out_ux"], "%08i.tiff" % int(i))
                save_uy = os.path.join(save_paths["out_uy"], "%08i.tiff" % int(i))
                ux = Image.fromarray(ux, mode="F")
                uy = Image.fromarray(uy, mode="F")
                ux.save(save_ux)
                uy.save(save_uy)

    def _load_batch_unsupervised(self, batch):
        in_ref = batch["Ref"].float().to(self.device)
        in_ref = torch.cat([in_ref, in_ref, in_ref], 1).to(self.device)
        in_def = batch["Def"].float().to(self.device)
        in_def = torch.cat([in_def, in_def, in_def], 1).to(self.device)
        inputs = torch.cat([in_ref, in_def], 1).to(self.device)

        return inputs
