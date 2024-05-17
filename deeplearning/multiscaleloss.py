import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import exp
from torch.autograd import Variable
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class EPE(torch.nn.Module):
    """
    End-Point-Error Loss Class. From StrainNet
    """

    def __init__(self, sparse: bool = False, mean: bool = True) -> None:
        """
        @param sparse: whether to exclude areas where the target flow is zero from
        the calculation
        @param mean: whether to return the mean or the sum of the EPE
        """
        super().__init__()
        self.sparse = sparse
        self.mean = mean

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate a mean norm-2 distance metric between two tensors.

        @param output: input tensor
        @param target: input tensor
        @return: scalar representing the EPE loss
        """
        epe_map = torch.norm(target - output, 2, 1)
        batch_size = epe_map.size(0)
        if self.sparse:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = (target[:, 0] == 0) & (target[:, 1] == 0)
            epe_map = epe_map[~mask]

        crop = [int(0.1 * epe_map.size()[1]), int(0.1 * epe_map.size()[2])]
        epe_map = epe_map[:, crop[0] : -crop[0], crop[1] : -crop[1]]
        if self.mean:
            return epe_map.mean()
        else:
            return epe_map.sum() / batch_size


class MultiscaleLoss(torch.nn.Module):
    """
    Multiscale loss evaluates a scalar-scaled loss at multiple network layer outputs (From StrainNet).

    Requires the StrainNet model.
    """

    def __init__(
        self,
        loss_fn: str = "EPE",
        loss_fn_args: dict = {"mean": False},
        weights: list = None,
        sparse: bool = False,
    ) -> None:
        """
        @param loss_fn: The loss function class
        @param sparse: whether to exclude areas where the target flow is zero from
        the calculation
        @param weights: weights to apply when summing the losses from each scale
        """
        super().__init__()

        if loss_fn_args is None:
            loss_fn_args = {}
        self.loss_fn = loss_function_dict[loss_fn](**loss_fn_args)

        if weights is None:
            self.weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
        else:
            self.weights = weights

        self.sparse = sparse

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss across multiple intermediate output tensors from the network.

        @param output: the output tensor from the network
        @param target: the target tensor
        @return: the loss value
        """
        if type(output) not in [tuple, list]:
            output = [output]
        assert len(self.weights) == len(
            output
        ), "Number of scales must match number of weights"

        loss = 0
        for output_scale, weight in zip(output, self.weights):
            loss += weight * self._one_scale(output_scale, target)
        return loss

    def _one_scale(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Resizes the target to the size of the output and calculates the loss.

        @param output: the output tensor from the network
        @param target: the target tensor
        @return: the loss value
        """
        b, _, h, w = output.size()

        if self.sparse:
            target_scaled = self._sparse_max_pool(target, (h, w))
        else:
            target_scaled = F.interpolate(target, (h, w), mode="area")

        return self.loss_fn(output, target_scaled)

    def _sparse_max_pool(self, input: torch.Tensor, size: tuple) -> torch.Tensor:
        """
        Applies adaptive sparse max pooling to the input tensor, scaling it to the
        given size, while retaining both positive and negative values.

        @param input: the input tensor
        @param size: the size of the output tensor
        @return: the output tensor
        """
        positive = (input > 0).float()
        negative = (input < 0).float()
        output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(
            -input * negative, size
        )
        return output


class realEPE(EPE):
    """
    Class for calculating the 'real' EPE loss, i.e. the EPE loss calculated after
    resizing the output to the size of the target.
    """

    def __init__(self, sparse: bool = False) -> None:
        super().__init__(sparse=sparse, mean=True)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, _, h, w = target.size()
        upsampled_output = F.interpolate(
            output, (h, w), mode="bilinear", align_corners=False
        )
        return super().forward(upsampled_output, target)


class SSIM(torch.nn.Module):
    """
    Structural Similarity Index (SSIM) Class
    """

    def __init__(
        self, window_size: int = 11, size_average: bool = True, map: bool = False
    ) -> None:
        """
        @param window_size: the size of windows
        @param size_average: whether to return the average SSIM across the entire image,
        or the SSIM values for each window
        @param map: whether to return the SSIM map
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = 1
        self.window = self._create_window()
        self.map = map

    def forward(self, img1, img2):
        (_, channels, _, _) = img1.size()

        if channels != self.channels or self.window.data.type() != img1.data.type():
            self.channels = channels
            self.window = self._create_window().type_as(img1)
            if img1.is_cuda:
                self.window = self.window.cuda(img1.get_device())

        return self._ssim(img1, img2)

    @staticmethod
    def _gaussian(window_size, sigma):
        """
        Create a
        """
        gauss = torch.Tensor(
            [
                exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def _create_window(self):
        _1D_window = self._gaussian(self.window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(
            _2D_window.expand(
                self.channels, 1, self.window_size, self.window_size
            ).contiguous()
        )
        return window

    def _ssim(self, img1, img2):
        mu1 = F.conv2d(
            img1, self.window, padding=self.window_size // 2, groups=self.channels
        )
        mu2 = F.conv2d(
            img2, self.window, padding=self.window_size // 2, groups=self.channels
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(
                img1 * img1,
                self.window,
                padding=self.window_size // 2,
                groups=self.channels,
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                img2 * img2,
                self.window,
                padding=self.window_size // 2,
                groups=self.channels,
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                img1 * img2,
                self.window,
                padding=self.window_size // 2,
                groups=self.channels,
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.map:
            return ssim_map
        elif self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1)


class SSIM_Loss(SSIM):
    """
    Class for calculating the SSIM loss, i.e. 1 - SSIM

    @param window_size: the size of windows
    @param size_average:
    @param map: whether to return the SSIM map
    """

    def __init__(
        self, window_size: int = 11, size_average: bool = True, map: bool = False
    ) -> None:
        super().__init__(window_size=window_size, size_average=size_average, map=map)

    def forward(self, img1, img2):
        return 1 - super().forward(img1, img2)


class NPCC(torch.nn.Module):
    """
    Negative Pearson Correlation Coefficient Loss
    """

    def __init__(self, eps=1e-6):
        """
        @param eps: small value to avoid division by 0
        """
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        batch_size = output.size(0)
        x = output - torch.mean(output)
        y = target - torch.mean(target)
        loss = torch.sum(x * y) / (
            torch.sqrt(torch.sum(x**2) + self.eps)
            * torch.sqrt(torch.sum(y**2) + self.eps)
        )
        loss = -1 * loss / batch_size
        return loss


class LPIPS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = LearnedPerceptualImagePatchSimilarity(net_type="alex")

    def _normalise(self, img):
        img = torch.clamp(img, min=-1, max=1)
        img = torch.cat((img, img, img), dim=1)
        return img

    def forward(self, img1, img2):
        # TODO: initialise all losses with device via kwargs at top level
        if self.loss.device != img1.device:
            self.loss.to(img1.device)

        if img1.size(dim=1) == 1:
            # imgs will be [B, 1, H, W] and need to be C=3
            # imgs need to be in -1, 1
            img1 = self._normalise(img1)
            img2 = self._normalise(img2)
            loss_lpips = self.loss(img1, img2)
        elif img1.size(dim=1) == 2:
            # Displacement vectors
            img1x = self._normalise(img1[:, 0:1, :, :])
            img1y = self._normalise(img1[:, 1:2, :, :])
            img2x = self._normalise(img2[:, 0:1, :, :])
            img2y = self._normalise(img2[:, 1:2, :, :])
            loss_lpips = 0.5 * (self.loss(img1x, img2x) + self.loss(img1y, img2y))
        else:
            loss_lpips = self.loss(img1, img2)

        return loss_lpips


class RewarpLoss(torch.nn.Module):
    def __init__(self, img_size, device, loss="NPCC"):
        super().__init__()

        self.img_size = img_size

        # convert to flow n, h, w, 2
        gx, gy = np.meshgrid(np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size))
        uxf = gx.reshape((1, img_size, img_size, 1))
        uyf = gy.reshape((1, img_size, img_size, 1))
        flow_grid_0 = np.concatenate((uxf, uyf), axis=3)
        self.flow_grid_0 = torch.from_numpy(flow_grid_0).float().to(device)

        self.loss = loss_function_dict[loss]()

    def forward(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        # deform input ref to def
        img_ref = input[:, 0:1, :, :]
        img_def = input[:, 3:4, :, :]

        flow = self.flow_grid_0 - 2 * output.permute((0, 2, 3, 1)) / self.img_size

        result = F.grid_sample(
            img_ref, flow, align_corners=True, padding_mode="border", mode="bicubic"
        )

        return self.loss(result, img_def)


# def calc_rewarp_pcc_forward(img_ref, img_def, ux, uy):
#     # rewarp - ux and uy were saved in transpose form because of xy indexing in plot vs save
#     img_rewarped = deform(img_ref, ux.transpose(), uy.transpose())
#     # calc pcc
#     pcc_val = calc_pcc(img_def, img_rewarped)
#     return pcc_val, img_rewarped


class ZeroDisplacementLoss(EPE):
    def __init__(self, sparse: bool = False) -> None:
        super().__init__(sparse=sparse, mean=True)

    def forward(self, output: torch.Tensor, target=None) -> torch.Tensor:
        target = torch.zeros_like(output)

        return super().forward(output, target)


class CombinedLoss(torch.nn.Module):
    """
    Class which combines other loss functions and returns their weighted sum
    """

    def __init__(
        self,
        losses: list,
        arguments: list | None = None,
        loss_weights: list | None = None,
    ):
        """
        :param losses: List of loss functions
        :param arguments: List of dictionaries containing arguments for each loss function
        :param loss_weights: List of weights to multiply each loss by
        """
        super(CombinedLoss, self).__init__()
        if arguments is None:
            self.losses = [loss() for loss in losses]
        else:
            assert len(losses) == len(arguments)
            self.losses = [loss(**arguments[i]) for i, loss in enumerate(losses)]

        if loss_weights is None:
            self.loss_weights = [1 / len(losses) for _ in losses]
        else:
            assert len(losses) == len(loss_weights)
            self.loss_weights = loss_weights

    def forward(self, output, target):
        results = [
            loss(output, target) * self.loss_weights[i]
            for i, loss in enumerate(self.losses)
        ]
        return sum(results)


# Dictionary of all loss functions mapping string to function class
# All loss functions must be nn.Module Classes
loss_function_dict = {
    "MultiscaleLoss": MultiscaleLoss,
    "realEPE": realEPE,
    "EPE": EPE,
    "SSIM_loss": SSIM_Loss,
    "SSIM": SSIM,
    "NPCC": NPCC,
    "LPIPS": LPIPS,
    "ZeroDisplacementLoss": ZeroDisplacementLoss,
    "RewarpLoss": RewarpLoss,
}


# Static functions below
# Non-torch loss implementations for validation
def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, map=False):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if map:
        return ssim_map
    elif size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)


def ssim(img1, img2, window_size=11, size_average=True, map=False):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, map)


def lpips_numpy(img1, img2, lpips_fn):
    img1t = torch.from_numpy(img1) / 8
    img2t = torch.from_numpy(img2) / 8
    img1t = img1t[None, None, :, :]
    img2t = img2t[None, None, :, :]
    loss_val = lpips_fn(img1t, img2t)
    return loss_val.detach().numpy()
