# -*- coding: utf-8 -*-
""" Trainer class for a generic strainnet type model
"""

import os
import time

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

# from helper.visualizer import calc_disp_phasor
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.utils import make_grid

import deeplearning.config as config_handler
import deeplearning.models as models
import deeplearning.transforms as tf

# import wandb
from deeplearning.datasets import DisplacementDataset
from deeplearning.multiscaleloss import CombinedLoss, loss_function_dict
from deeplearning.utils import AverageMeter, save_checkpoint


class Trainer:
    def __init__(self, config, resume=True):
        """Trainer class for deep learning

        The data and parameters should be passed via the config dictionary. See test_config
        """
        cudnn.benchmark = (
            True  # precalculates and optimises algorithms based on network shape?
        )
        preload_data = True

        self.n_iter = None
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("Warning: CUDA not available, using CPU")
            self.device = torch.device("cpu")

        # TODO: Optional: Prepare folders from config - needs validation and overwrite control
        save_dir = self.config["dirs"]["output"]
        self.save_paths = {
            "root": save_dir,
            "model": os.path.join(save_dir, "saved_model"),
            "log": os.path.join(save_dir, "log"),
        }
        for key, val in self.save_paths.items():
            if not os.path.exists(val):
                os.makedirs(val)

        # TensorboardX writer - launch via cmd: tensorboard --logdir=<path to train>
        self.writer = SummaryWriter(self.save_paths["log"])

        # wandb
        # self.run = wandb.init(
        #     project="displacement-learning",
        #     name=self.config["run_name"],
        #     group=self.config["group"],
        #     entity="biophotonics-st-andrews",
        #     config=self.config,
        #     reinit=True,
        # )

        # Save config dict for reference
        config_handler.save(
            os.path.join(self.save_paths["root"], "config_run.yml"), self.config
        )

        # Dataloaders
        # - TODO: Optional: Prepare transforms based on config
        self.transforms_train = [
            tf.Normalization(**config["transforms"]["normalisation"]),
            tf.SimultaneousCrop(img_size=self.config["img_size"]),
            tf.SimultaneousDiscreteRotate(),
            # tf.AddGaussNoise(noise_std_pc=self.config['transforms']['noise'])
            tf.AddPoissonNoise(noise_std_pc=self.config["transforms"]["noise"]),
        ]
        self.transforms_val = [
            tf.Normalization(**config["transforms"]["normalisation"]),
            tf.SimultaneousCrop(img_size=self.config["img_size"]),
            # tf.AddGaussNoise(noise_std_pc=self.config['transforms']['noise'])
            tf.AddPoissonNoise(noise_std_pc=self.config["transforms"]["noise"]),
        ]
        self.transforms_val_image = [
            tf.Normalization(**config["transforms"]["normalisation"]),
            tf.SimultaneousCenterCrop(img_size=self.config["img_size"]),
            # tf.AddGaussNoise(noise_std_pc=self.config['transforms']['noise'])
            tf.AddPoissonNoise(noise_std_pc=self.config["transforms"]["noise"]),
        ]

        # - TODO: Optional: Needs assertion that data exists
        print(f"Using training data from {self.config['dirs']['train']}")
        dataset_train = DisplacementDataset(
            self.config["dirs"]["train"],
            transforms_=self.transforms_train,
            preload=preload_data,
        )
        print(f"Using validation data from {self.config['dirs']['val']}")
        dataset_val = DisplacementDataset(
            self.config["dirs"]["val"],
            transforms_=self.transforms_val,
            preload=preload_data,
        )
        print(f"Using validation images from {self.config['dirs']['val']}")
        dataset_val_image = DisplacementDataset(
            self.config["dirs"]["val"],
            transforms_=self.transforms_val_image,
            preload=preload_data,
        )
        self.dataloader_train = DataLoader(
            dataset=dataset_train, batch_size=self.config["batch_size"]
        )
        self.dataloader_val = DataLoader(
            dataset=dataset_val, batch_size=self.config["batch_size"]
        )
        self.dataloader_val_image = DataLoader(dataset=dataset_val_image, batch_size=1)

        # Load model
        print(f"Using model {self.config['model']}")
        self.model = models.__dict__[self.config["model"]]()

        # Check if model exists and reload
        self.resume_epoch = 0
        model_path = os.path.join(self.save_paths["model"], "checkpoint.pt")
        if os.path.exists(model_path) and resume:
            network_data = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(network_data["state_dict"])
            self.resume_epoch = network_data["epoch"]
            print(f"Loading existing model from epoch {self.resume_epoch}")

        # self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # watch
        # wandb.watch(self.model)

        try:
            param_groups = [
                {"params": self.model.bias_parameters(), "weight_decay": 0},
                {"params": self.model.weight_parameters(), "weight_decay": 4e-4},
            ]  # Are these needed?
        except:
            param_groups = self.model.parameters()

        print(f"Using optimizer Adam")
        self.optimizer = torch.optim.Adam(
            params=param_groups,
            lr=self.config["lr"],
            betas=(self.config["b1"], self.config["b2"]),
        )

        # Initialise loss functions and metrics
        if isinstance(config["loss"], list):
            print(f"Using combined loss functions {self.config['loss']}")
            loss_args = config["loss_args"] if config["loss_args"] else None
            loss_weights = config["loss_weights"] if config["loss_weights"] else None
            self.loss_function = CombinedLoss(
                [loss_function_dict[loss] for loss in config["loss"]],
                arguments=loss_args,
                loss_weights=loss_weights,
            )
        else:
            print(f"Using loss function {self.config['loss']}")
            if config["loss_args"]:
                print(f"Using loss function arguments {self.config['loss_args']}")
                self.loss_function = loss_function_dict[config["loss"]](
                    **config["loss_args"]
                )
            else:
                self.loss_function = loss_function_dict[config["loss"]]()

        self.realEPE = loss_function_dict["realEPE"]()
        self.SSIM = loss_function_dict["SSIM"]()
        self.Rewarp = loss_function_dict["RewarpLoss"](
            img_size=self.config["img_size"][0],
            device=self.device,
            loss=self.config["rewarp_loss"],
        )
        self.LPIPS = loss_function_dict["LPIPS"]()

        if self.config["scheduler"]["use"]:
            print(
                f"Using scheduler with milestones {self.config['scheduler']['milestones']} and gamma {self.config['scheduler']['gamma']}"
            )
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config["scheduler"]["milestones"],
                gamma=self.config["scheduler"]["gamma"],
            )
        else:
            self.scheduler = None

    def train(self):
        print("Starting training...")
        self.n_iter = 0
        loss_val_best = 0
        for epoch in range(self.resume_epoch, self.config["epochs"]):
            self._train_epoch(epoch)
            with torch.no_grad():
                loss_val = self._validate_epoch(epoch)

            isbest = False
            if (
                epoch > self.config["epochs"] // 2
            ):  # save best in the last half of training
                if loss_val_best == 0 or (loss_val < loss_val_best):
                    isbest = True
                    loss_val_best = loss_val

            self._save_checkpoint(epoch, isbest=isbest)
            self._write_image(epoch)
            self.writer.flush()

        # self.run.finish()

    def _write_image(self, epoch):
        # Try with a single image
        self.model.eval()
        batch = next(iter(self.dataloader_val_image))
        target, inputs = self._load_batch(batch)
        output = self.model(inputs)

        img_input = inputs[0,].detach().cpu()
        img_target = target[0,].detach().cpu()
        img_output = output[0,].detach().cpu()

        img_ref = img_input[
            0,
        ]
        img_def = img_input[
            3,
        ]

        crop_size = 64
        crop = transforms.CenterCrop(
            [
                self.config["img_size"][0] - crop_size,
                self.config["img_size"][1] - crop_size,
            ]
        )

        img = make_grid(
            [
                crop(img_ref.unsqueeze(0)),
                crop(img_target[0,].unsqueeze(0)),
                crop(img_target[1,].unsqueeze(0)),
                crop(img_def.unsqueeze(0)),
                crop(img_output[0,].unsqueeze(0)),
                crop(img_output[1,].unsqueeze(0)),
            ],
            nrow=3,
            scale_each=True,
            normalize=True,
        )
        # TODO: manual scaling of each image set - for some reason unscaled images are wrapped mod 1?

        self.writer.add_image("val_image", img, epoch)

        # images = wandb.Image(img)
        # wandb.log({"examples": images, "epoch": epoch})

    def _load_batch(self, batch):
        target_ux = batch["Dispx"].to(self.device)
        target_uy = batch["Dispy"].to(self.device)
        target = torch.cat([target_ux, target_uy], 1).to(self.device)

        in_ref = batch["Ref"].float().to(self.device)
        in_ref = torch.cat([in_ref, in_ref, in_ref], 1).to(self.device)

        in_def = batch["Def"].float().to(self.device)
        in_def = torch.cat([in_def, in_def, in_def], 1).to(self.device)
        inputs = torch.cat([in_ref, in_def], 1).to(self.device)

        return target, inputs

    def _save_checkpoint(self, epoch, isbest=False):
        state_dict = {
            "epoch": epoch + 1,
            "arch": self.config["model"],
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_EPE": None,
            "div_flow": None,
        }
        save_checkpoint(state_dict, isbest, self.save_paths["model"])
        if not (epoch % self.config["save_period"]):
            save_checkpoint(
                state_dict,
                0,
                self.save_paths["model"],
                filename="checkpoint_{0}.pt".format(epoch),
            )

    def _train_epoch(self, epoch):
        metric_loss = AverageMeter()
        metric_loss_epe = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        if self.scheduler:
            self.scheduler.step()

        self.model.train()
        time_last = time.time()
        for i, batch in enumerate(self.dataloader_train):
            data_time.update(time.time() - time_last)

            target, inputs = self._load_batch(batch)
            output = self.model(inputs)

            # TODO: verify compatability with all loss functions, add argument passing from config
            # Multiscale loss uses weighted distance loss for last few layers (See strainnet paper)

            if self.config["loss"] == "MultiscaleLoss" or self.config["loss"] == [
                "MultiscaleLoss"
            ]:
                if self.config["model"] == "StrainNetF":
                    loss = self.loss_function(output, target)
                    loss_epe = self.realEPE(output[0], target)
                else:
                    raise NotImplementedError(
                        "Multiscale loss has only been implemented for StrainNetF"
                    )
            else:
                if self.config["model"] == "StrainNetF":
                    loss = self.loss_function(output[0], target)
                    loss_epe = self.realEPE(output[0], target)
                else:
                    loss = self.loss_function(output, target)
                    loss_epe = self.realEPE(output, target)

            # 'real' loss is the distance metric for final output vs target (used for logging)

            # Unsupervised rewarp loss
            if self.config["use_rewarp"]:
                loss += self.Rewarp(output, inputs) * self.config["rewarp_weight"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if self.scheduler:
            #     self.scheduler.step()

            metric_loss.update(loss.item(), target.size(0))
            metric_loss_epe.update(loss_epe.item(), target.size(0))
            self.writer.add_scalar("train/train_loss", loss.item(), self.n_iter)
            # wandb.log({"train/train_loss": loss.item(), "epoch": epoch, "batch": self.n_iter})
            self.n_iter += 1

            batch_time.update(time.time() - time_last)
            time_last = time.time()
            if i % self.config["update_period"] == 0:
                print(
                    "Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}".format(
                        epoch,
                        i,
                        len(self.dataloader_train),
                        batch_time,
                        data_time,
                        metric_loss,
                        metric_loss_epe,
                    )
                )

        self.writer.add_scalar("train/mean_EPE", metric_loss_epe.avg, epoch)
        # wandb.log({"train/train_loss": metric_loss.avg, "epoch": epoch})
        # wandb.log({"train/mean_EPE": metric_loss_epe.avg, "epoch": epoch})
        return metric_loss.avg, metric_loss_epe.avg

    def _validate_epoch(self, epoch):
        metric_ssim = AverageMeter()
        metric_loss_epe = AverageMeter()
        metric_rewarp = AverageMeter()
        metric_lpips = AverageMeter()
        batch_time = AverageMeter()

        self.model.eval()
        time_last = time.time()
        for i, batch in enumerate(self.dataloader_val):
            target, inputs = self._load_batch(batch)
            output = self.model(inputs)

            loss_ssim = self.SSIM(output, target)
            loss_epe = self.realEPE(output, target)
            loss_lpips = self.LPIPS(output, target)
            metric_loss_epe.update(loss_epe.item(), target.size(0))
            metric_ssim.update(loss_ssim.item(), target.size(0))
            metric_lpips.update(loss_lpips.item(), target.size(0))
            # if self.config["use_rewarp"]:
            # always monitor loss
            loss_rewarp = self.Rewarp(output, inputs)
            metric_rewarp.update(loss_rewarp.item(), target.size(0))

            batch_time.update(time.time() - time_last)
            time_last = time.time()
            if i % self.config["update_period"] == 0:
                print(
                    "Val: [{0}/{1}]\t Time {2}\t EPE {3}".format(
                        i, len(self.dataloader_val), batch_time, metric_loss_epe
                    )
                )

        self.writer.add_scalar("val/mean_EPE", metric_loss_epe.avg, epoch)
        self.writer.add_scalar("val/mean_SSIM", metric_ssim.avg, epoch)
        self.writer.add_scalar("val/mean_LPIPS", metric_lpips.avg, epoch)
        # wandb.log({"val/mean_EPE": metric_loss_epe.avg, "epoch": epoch})
        # wandb.log({"val/mean_SSIM": metric_ssim.avg, "epoch": epoch})
        # wandb.log({"val/mean_LPIPS": metric_lpips.avg, "epoch": epoch})
        # if self.config["use_rewarp"]:
        self.writer.add_scalar("val/mean_Rewarp", metric_rewarp.avg, epoch)
        # wandb.log({"val/mean_Rewarp": metric_rewarp.avg, "epoch": epoch})

        return metric_loss_epe.avg
