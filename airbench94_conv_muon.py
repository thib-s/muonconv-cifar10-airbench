"""
airbench94_muon.py
Runs in 2.59 seconds on a 400W NVIDIA A100 using torch==2.4.1
Attains 94.01 mean accuracy (n=200 trials)
Descends from https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""

#############################################
#                  Setup                    #
#############################################

import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()
import argparse
import uuid
from math import ceil, sqrt

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

#############################################
#               Muon optimizer              #
#############################################


import torch
import torch.nn.functional as F


def compute_delattre2023(X, n=None, n_iter=3):
    """Estimate spectral norm of convolutional layer with Delattre2023.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer for circular padding using [Section.3, Algo. 3] Delattre2023.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, k, k)
        Convolutional filter.
    n : None | int, default=None
        Size of input image. If None, n is set equal to k.
    n_iter : int, default=4
        Number of iterations.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.
    """
    cout, cin, k, _ = X.shape
    if n is None:
        n = k
    if cin > cout:
        X = X.transpose(0, 1)
        cin, cout = cout, cin

    crossed_term = torch.fft.rfft2(X, s=(n, n)).reshape(cout, cin, -1).permute(2, 0, 1)
    inverse_power = 1
    log_curr_norm = torch.zeros(crossed_term.shape[0]).to(crossed_term.device)
    for _ in range(n_iter):
        norm_crossed_term = crossed_term.norm(dim=(1, 2))
        crossed_term /= norm_crossed_term.reshape(-1, 1, 1)
        log_curr_norm = 2 * log_curr_norm + norm_crossed_term.log()
        crossed_term = torch.bmm(crossed_term.conj().transpose(1, 2), crossed_term)
        inverse_power /= 2
    sigma = (
        crossed_term.norm(dim=(1, 2)).pow(inverse_power)
        * ((2 * inverse_power * log_curr_norm).exp())
    ).max()

    return sigma


def compute_spectral_rescaling_conv(kernel, n_iter=1):
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least equal to 1, got {n_iter}")
    effective_iter = 0
    kkt = kernel
    log_curr_norm = 0
    for _ in range(n_iter):
        padding = kkt.shape[-1] - 1
        kkt_norm = kkt.norm().detach()
        kkt = kkt / kkt_norm
        log_curr_norm = 2 * (log_curr_norm + kkt_norm.log())
        kkt = F.conv2d(kkt, kkt, padding=padding)
        effective_iter += 1
    inverse_power = 2 ** (-effective_iter)
    t = torch.abs(kkt)
    t = t.sum(dim=(1, 2, 3)).pow(inverse_power)
    norm = torch.exp(log_curr_norm * inverse_power)
    t = t * norm
    return t


def orthogonalize_kernel_beta(
    ker: torch.Tensor,
    beta_init: float = 2.0,
    beta_end: float = 0.5,
    num_iters: int = 10,
    damp=0.98,
    epsilon: float = 1e-1,
) -> torch.Tensor:
    """
    Orthogonalizes a conv2d kernel with a bjorck-based update, and inlines an AOL
    1-Lipschitz (L2) per-input-channel rescale. Standalone, copy-pastable.

    Args:
        ker: conv2d kernel [outC, inC, kh, kw] (PyTorch layout)
        beta_init: starting beta value (inclusive)
        beta_end: ending beta value (inclusive)
        num_iters: number of iterations
        epsilon: small positive constant for AOL rescale numerical stability

    Returns:
        torch.Tensor with same shape as `ker`
    """
    assert ker.ndim == 4, "ker must be [outC, inC, kh, kw]"
    co, ci, kh, kw = ker.shape
    ph, pw = kh - 1, kw - 1

    # if co <= ci:
    #     ker = torch.concat((ker, torch.randn(1, ci, kw, kh, device=ker.device, dtype=ker.dtype) * 0.05), dim=0)
    # else :
    #     ker = torch.concat((ker, torch.randn(co, 1, kw, kh, device=ker.device, dtype=ker.dtype) * 0.05), dim=1)

    # # ---- AOL per-input-channel rescale (1-Lipschitz in L2) ----
    # # This ensure that the operator norm of the convolution is < 1.0 ensuring
    # # stability of the bjorck algorithm
    # # Use a stride-only view instead of permute+contiguous
    # x = ker.transpose(0, 1)  # [ci, co, kh, kw] (view)
    # w = x                    # same tensor/layout

    # # Pad for full correlation and convolve
    # x_pad = F.pad(x, (pw, pw, ph, ph))
    # v = F.conv2d(x_pad, w, bias=None, stride=1, padding=0)  # [ci, ci, 2kh-1, 2kw-1]

    # # Sum abs over channel+spatial, then compute inverse sqrt
    # lipschitz_bounds_sq = v.abs().sum(dim=(1, 2, 3))        # [ci]
    # rescale = (lipschitz_bounds_sq + epsilon).pow(-0.5)     # [ci]
    # ker = ker * rescale.view(1, -1, 1, 1)
    ## AOL tested: not good
    ## frobenius much better
    # ker = ker / (torch.sqrt(torch.sum(torch.square(ker))) + epsilon) # global rescale to ensure stability of bjorck algorithm, and to keep the scale of the updates consistent across different random initializations
    ## spectral give current best results
    ker = ker / (compute_spectral_rescaling_conv(ker, n_iter=1).max() + epsilon)
    ## RKO might be worth to try
    # ker = newton_schulz(ker.reshape(co, -1)).reshape_as(ker) / sqrt(kw + kh) # orthogonalize the kernel to improve conditioning of the bjorck algorithm

    # ---- Bjork algorithm ----
    # For dense layer bjorck consists in recursing the following scheme:
    # W <- (1 + \beta) W - beta W@W^T@W
    # but we cannot do that with convolution, to do this we would need to extract the
    # large toeplitz matrices! So the trick is to use block convolution to compute this
    # scheme implicitely using block convolutioon.
    # In orther words we compute how the iterate would affect the kernel (and not the
    # toepliz matrix )
    # K = (1 + \beta) K - beta K.bc.K^T.bc.K
    # where bc is the block conv and ^T the *kernel* transposition
    # The problem is that K.bc.K^T constructs a kernel larger than K, so we crop it
    # to keep its original size. In practice, this acts as an alternate projection onto
    # fixed kernel supports and orthogonal convolutions.

    # We change beta from large values at the beginning to small values at the end.
    betas = torch.linspace(
        beta_init, beta_end, steps=max(1, num_iters), dtype=ker.dtype, device=ker.device
    )

    for beta in betas:
        # kk = computes K.bc.K^T
        # Shapes: ker [C_in=co, C_out=ci, kh, kw]
        # Output: input=kk [N=co, C_in=co, 2kh-1, 2kw-1]
        kk = F.conv2d(ker, ker, padding=(ph, pw))

        # kkk = k.bc.k^t.bc.k -> full conv of kk with ker -> conv_transpose2d with padding=0
        # (conv transpose avoids the use of ker.transpose(0,1).reverse(2,3) to transpose kernel)
        # Shapes: input=kk [N=co, C_in=co, 2kh-1, 2kw-1], weight=ker [C_in=co, C_out=ci, kh, kw]
        # Output: [N=co, C_out=ci, 3kh-2, 3kw-2]
        kkk = F.conv_transpose2d(kk, ker, padding=(ph, pw))
        # Beta update (with damping)
        ker = (1.0 + beta) * damp * ker - beta * damp * kkk

    return ker  # [:co, :ci, :, :]


# def orthogonalize_kernel_beta(
#     ker: torch.Tensor,
#     beta_init: float = 0.5,
#     beta_end: float = 0.5,
#     num_iters: int = 10,
#     damp = 0.99,
#     epsilon: float = 0.01,
#     padding: int = 1,
# ) -> torch.Tensor:
#     """
#     Orthogonalizes a conv2d kernel with a bjorck-based update, and inlines an AOL
#     1-Lipschitz (L2) per-input-channel rescale.

#     Args:
#         ker: conv2d kernel [outC, inC, kh, kw]
#         beta_init: starting beta value
#         beta_end: ending beta value
#         num_iters: number of iterations
#         damp: damping factor
#         epsilon: small positive constant for AOL rescale stability
#         padding: number of pixels to zero-pad the kernel before the loop

#     Returns:
#         torch.Tensor with same shape as `ker` (original shape)
#     """
#     assert ker.ndim == 4, "ker must be [outC, inC, kh, kw]"
#     co, ci, kh, kw = ker.shape

#     # ---- Delattra per-input-channel rescale ----
#     ker = ker / (compute_spectral_rescaling_conv(ker, n_iter=2).max() + epsilon)

#     # ---- Padding ----
#     if padding > 0:
#         ker = F.pad(ker, (padding, padding, padding, padding))

#     ker_o = ker.clone()

#     # Update shapes for Bjork logic
#     _, _, current_kh, current_kw = ker.shape
#     ph, pw = current_kh - 1, current_kw - 1

#     betas = torch.linspace(
#         beta_init, beta_end, steps=max(1, num_iters),
#         dtype=ker.dtype, device=ker.device
#     )

#     for beta in betas:
#         kk = F.conv2d(ker, ker, padding=(ph, pw))
#         kkk = F.conv_transpose2d(kk, ker, padding=(ph, pw))

#         # Bjork update
#         ker = (1.0 + beta) * ker - beta * kkk

#         # Divide padded values by 2 if padding was applied
#         if padding > 0:
#             # mask out the central part to identify padding
#             mask = torch.ones_like(ker)
#             mask[:, :, padding:-padding, padding:-padding] = 0
#             ker = ker * (1.0 - mask * 0.5)

#         ker = damp * ker + (1- damp) * ker_o

#     # ---- Crop back to original size ----
#     if padding > 0:
#         ker = ker[:, :, padding:-padding, padding:-padding]

#     return ker


@torch.compile
def newton_schulz(
    G, iter=5, precondition=False, epsilon: float = 1e-7, dtype=torch.bfloat16
):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ][-iter:]
    X = G.to(dtype=dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    if not precondition:
        X /= X.norm(dim=(-2, -1), keepdim=True) + epsilon

    # Perform the NS iterations
    for i, (a, b, c) in enumerate(ns_consts):
        A = X @ X.mT
        if precondition and i == 0:
            s = torch.rsqrt(
                torch.clamp_min(A.abs().sum(dim=-1, keepdim=False), min=epsilon)
            )
            X = X * s.unsqueeze(-1)
            A = A * s.unsqueeze(-1) * s.unsqueeze(-2)

        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        nesterov=False,
        orthogonalize_beta_init=2.0,
        orthogonalize_beta_end=0.5,
        orthogonalize_num_iters=10,
        orthogonalize_damp=0.98,
        orthogonalize_epsilon=1e-1,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            orthogonalize_beta_init=orthogonalize_beta_init,
            orthogonalize_beta_end=orthogonalize_beta_end,
            orthogonalize_num_iters=orthogonalize_num_iters,
            orthogonalize_damp=orthogonalize_damp,
            orthogonalize_epsilon=orthogonalize_epsilon,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data) ** 0.5 / p.data.norm())  # normalize the weight
                if len(g.shape) == 4:
                    update = orthogonalize_kernel_beta(
                        g,
                        beta_init=group.get("orthogonalize_beta_init", 2.0),
                        beta_end=group.get("orthogonalize_beta_end", 0.5),
                        num_iters=group.get("orthogonalize_num_iters", 10),
                        damp=group.get("orthogonalize_damp", 0.98),
                        epsilon=group.get("orthogonalize_epsilon", 1e-1),
                    )
                    cout, cin, kh, kw = g.shape
                    update = update * sqrt(cout / cin)
                elif (len(g.shape) == 2) or (len(g.shape) == 3):
                    update = newton_schulz(g)  # whiten the update
                    update = update * sqrt(g.shape[-0] / update.shape[-1])
                else:
                    raise NotImplementedError(
                        "Muon only supports 2D, 3D, and 4D parameters"
                    )
                p.data.add_(update, alpha=-lr)  # take a step


#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty(
        (len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype
    )
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[
                    mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size
                ]
    else:
        images_tmp = torch.empty(
            (len(images), 3, crop_size, crop_size + 2 * r),
            device=images.device,
            dtype=images.dtype,
        )
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s : r + s + crop_size]
    return images_out


class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save(
                {"images": images, "labels": labels, "classes": dset.classes}, data_path
            )

        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = (
            data["images"],
            data["labels"],
            data["classes"],
        )
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (
            (self.images.half() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = (
            {}
        )  # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(
            len(images), device=images.device
        )
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])


#############################################
#            Network Definition             #
#############################################


# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(
            3, whiten_width, whiten_kernel_size, padding=0, bias=True
        )
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = (
            train_images.unfold(2, h, 1)
            .unfold(3, w, 1)
            .transpose(1, 3)
            .reshape(-1, c, h, w)
            .float()
        )
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(
            eigenvalues.view(-1, 1, 1, 1) + eps
        )
        self.whiten.weight.data[:] = torch.cat(
            (eigenvectors_scaled, -eigenvectors_scaled)
        )

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


############################################
#                 Logging                  #
############################################


def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-" * len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-" * len(print_string))


logging_columns_list = [
    "run   ",
    "epoch",
    "train_acc",
    "val_acc",
    "tta_val_acc",
    "time_seconds",
]


def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)


############################################
#               Evaluation                 #
############################################


def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,) * 4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [
            infer_mirror(inputs_translate, net)
            for inputs_translate in inputs_translate_list
        ]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat(
            [infer_fn(inputs, model) for inputs in test_images.split(2000)]
        )


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


############################################
#                Training                  #
############################################

# DEFAULT_OPTIMIZER_CONFIG = {
#     "bias_lr": 0.053,
#     "head_lr": 0.67,
#     "weight_decay_scale": 1e-6,
#     "sgd_momentum": 0.85,
#     "muon_lr": 0.52,
#     "muon_momentum": 0.6,
#     "orthogonalize_beta_init": 0.5,
#     "orthogonalize_beta_end": 0.5,
#     "orthogonalize_num_iters": 12,
#     "orthogonalize_damp": 0.91,
#     "orthogonalize_epsilon": 0.1,
# }

DEFAULT_OPTIMIZER_CONFIG = {
    "bias_lr": 0.053,
    "head_lr": 0.67,
    "weight_decay_scale": 1e-6,
    "sgd_momentum": 0.85,
    "muon_lr": 0.52,
    "muon_momentum": 0.6,
    "orthogonalize_beta_init": 0.5,
    "orthogonalize_beta_end": 0.5,
    "orthogonalize_num_iters": 12,
    "orthogonalize_damp": 0.91,
    "orthogonalize_epsilon": 0.1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CIFAR10 Airbench with optional W&B sweep-configurable optimizer hyperparameters."
    )
    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of measured runs after warmup."
    )
    parser.add_argument(
        "--batch-size", type=int, default=2000, help="Training batch size."
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="W&B project name."
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity/team."
    )
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        help="W&B mode, e.g. online/offline/disabled.",
    )
    parser.add_argument(
        "--bias-lr",
        type=float,
        default=None,
        help="Learning rate for whitening bias and norm biases.",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="Learning rate for classification head.",
    )
    parser.add_argument(
        "--weight-decay-scale",
        type=float,
        default=None,
        help="Scale so actual weight decay is weight_decay_scale * batch_size.",
    )
    parser.add_argument(
        "--sgd-momentum",
        type=float,
        default=None,
        help="Momentum for SGD parameter groups.",
    )
    parser.add_argument(
        "--muon-lr", type=float, default=None, help="Learning rate for Muon."
    )
    parser.add_argument(
        "--muon-momentum", type=float, default=None, help="Momentum for Muon."
    )
    parser.add_argument(
        "--orthogonalize-beta-init",
        type=float,
        default=None,
        help="Initial beta used by conv orthogonalization in Muon.",
    )
    parser.add_argument(
        "--orthogonalize-beta-end",
        type=float,
        default=None,
        help="Final beta used by conv orthogonalization in Muon.",
    )
    parser.add_argument(
        "--orthogonalize-num-iters",
        type=int,
        default=None,
        help="Number of iterations for conv orthogonalization in Muon.",
    )
    parser.add_argument(
        "--orthogonalize-damp",
        type=float,
        default=None,
        help="Damping used by conv orthogonalization in Muon.",
    )
    parser.add_argument(
        "--orthogonalize-epsilon",
        type=float,
        default=None,
        help="Epsilon used by conv orthogonalization in Muon.",
    )
    return parser.parse_args()


def build_optimizer_config(args, wandb_run):
    optimizer_config = dict(DEFAULT_OPTIMIZER_CONFIG)
    cli_overrides = {
        "bias_lr": args.bias_lr,
        "head_lr": args.head_lr,
        "weight_decay_scale": args.weight_decay_scale,
        "sgd_momentum": args.sgd_momentum,
        "muon_lr": args.muon_lr,
        "muon_momentum": args.muon_momentum,
        "orthogonalize_beta_init": args.orthogonalize_beta_init,
        "orthogonalize_beta_end": args.orthogonalize_beta_end,
        "orthogonalize_num_iters": args.orthogonalize_num_iters,
        "orthogonalize_damp": args.orthogonalize_damp,
        "orthogonalize_epsilon": args.orthogonalize_epsilon,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            optimizer_config[key] = value

    if wandb_run is not None:
        for key in optimizer_config:
            if key in wandb_run.config:
                optimizer_config[key] = wandb_run.config[key]
        wandb_run.config.update(optimizer_config, allow_val_change=True)

    optimizer_config["orthogonalize_num_iters"] = int(
        optimizer_config["orthogonalize_num_iters"]
    )
    optimizer_config["orthogonalize_epsilon"] = float(
        optimizer_config["orthogonalize_epsilon"]
    )
    return optimizer_config


def main(run, model, optimizer_config, batch_size=2000, wandb_run=None):

    is_warmup = run == "warmup"
    bias_lr = optimizer_config["bias_lr"]
    head_lr = optimizer_config["head_lr"]
    wd = optimizer_config["weight_decay_scale"] * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device
        )
    total_train_steps = ceil(6 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers and learning rate schedulers
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs,
        momentum=optimizer_config["sgd_momentum"],
        nesterov=True,
        fused=True,
    )
    optimizer2 = Muon(
        filter_params,
        lr=optimizer_config["muon_lr"],
        momentum=optimizer_config["muon_momentum"],
        nesterov=True,
        orthogonalize_beta_init=optimizer_config["orthogonalize_beta_init"],
        orthogonalize_beta_end=optimizer_config["orthogonalize_beta_end"],
        orthogonalize_num_iters=optimizer_config["orthogonalize_num_iters"],
        orthogonalize_damp=optimizer_config["orthogonalize_damp"],
        orthogonalize_epsilon=optimizer_config["orthogonalize_epsilon"],
    )
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    # Initialize the whitening layer using training images
    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for epoch in range(ceil(total_train_steps / len(train_loader))):

        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(
                outputs, labels, label_smoothing=0.2, reduction="sum"
            ).backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        if wandb_run is not None and not is_warmup:
            wandb_run.log(
                {
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "time_seconds": time_seconds,
                }
            )
        run = None  # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)
    if wandb_run is not None and not is_warmup:
        wandb_run.log(
            {
                "tta_val_acc": tta_val_acc,
                "time_seconds": time_seconds,
            }
        )

    return tta_val_acc


if __name__ == "__main__":
    args = parse_args()

    wandb_run = None
    use_wandb = args.wandb or ("WANDB_SWEEP_ID" in os.environ)
    if use_wandb:
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "W&B is enabled but `wandb` is not installed. Install it with `pip install wandb`."
            ) from exc
        wandb_kwargs = {}
        if args.wandb_project is not None:
            wandb_kwargs["project"] = args.wandb_project
        if args.wandb_entity is not None:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_name is not None:
            wandb_kwargs["name"] = args.wandb_name
        if args.wandb_mode is not None:
            wandb_kwargs["mode"] = args.wandb_mode
        wandb_run = wandb.init(config=dict(DEFAULT_OPTIMIZER_CONFIG), **wandb_kwargs)

    optimizer_config = build_optimizer_config(args, wandb_run)

    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")

    print_columns(logging_columns_list, is_head=True)
    main(
        "warmup",
        model,
        optimizer_config=optimizer_config,
        batch_size=args.batch_size,
        wandb_run=wandb_run,
    )
    accs = torch.tensor(
        [
            main(
                run,
                model,
                optimizer_config=optimizer_config,
                batch_size=args.batch_size,
                wandb_run=wandb_run,
            )
            for run in range(args.num_runs)
        ]
    )
    accs_mean = accs.mean()
    accs_std = accs.std(unbiased=False)
    print("Mean: %.4f    Std: %.4f" % (accs_mean, accs_std))

    if wandb_run is not None:
        wandb_run.summary["tta_val_acc_mean"] = accs_mean.item()
        wandb_run.summary["tta_val_acc_std"] = accs_std.item()
        wandb_run.summary["num_runs"] = int(args.num_runs)

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, accs=accs, optimizer_config=optimizer_config), log_path)
    print(os.path.abspath(log_path))

    if wandb_run is not None:
        wandb_run.finish()
