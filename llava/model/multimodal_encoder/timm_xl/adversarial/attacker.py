""" PGD Attack in PyTorch

Hacked together by / Copyright 2024, Zeyu Wang
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import logging
_logger = logging.getLogger(__name__)


class NoOpAttacker():
    
    def attack(self, image, label, model, criterion):
        return image, label


class PGDAttacker():
    def __init__(self, num_iter, epsilon, step_size, dev_env,
                 mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
                 grad_scaler=None):
        self.num_iter = num_iter
        self.epsilons = epsilon
        self.step_sizes = step_size
        self.dev_env = dev_env
        self.cast_dtype = dev_env.cast_dtype
        self.mean = torch.tensor(mean, dtype=self.cast_dtype)
        self.std = torch.tensor(std, dtype=self.cast_dtype)
        self.mean, self.std = self.dev_env.to_device(self.mean, self.std)
        self.get_pixel_range()
        self.get_image_scale()
        _logger.info(f'using pgd attack num_iter {num_iter} epsilon {epsilon} step_size {step_size}')

        self.grad_scaler = grad_scaler


    def set_mean_std(self, mean, std):
        self.mean = torch.tensor(mean, dtype=self.cast_dtype)
        self.std = torch.tensor(std, dtype=self.cast_dtype)
        self.mean, self.std = self.dev_env.to_device(self.mean, self.std)
        self.get_pixel_range()
        self.get_image_scale()

    def get_pixel_range(self):
        # assume 0-255 original pixel / 255
        lower_bounds = torch.zeros(3, dtype=self.cast_dtype)
        upper_bounds = torch.ones(3, dtype=self.cast_dtype)
        lower_bounds, upper_bounds = self.dev_env.to_device(lower_bounds, upper_bounds)
        self.lower_bounds = (lower_bounds - self.mean) / self.std
        self.upper_bounds = (upper_bounds - self.mean) / self.std

    def get_image_scale(self):
        # note that this scale means upper_bound and lower_bound calibrated to (0,255). it is different from grad_scaler in amp!
        self.image_scales = (self.upper_bounds - self.lower_bounds) / 255.
        self.scaled_epsilons = self.epsilons * self.image_scales
        self.scaled_step_sizes = self.step_sizes * self.image_scales

    def set_grad_scaler(self, grad_scaler):
        self.grad_scaler = grad_scaler

    def attack(self, image_clean, label, model, criterion):
        target_label = label

        # assume rgb input
        lower_bounds = []
        upper_bounds = []
        for i in range(3):
            lower_bounds.append(
                torch.clamp(image_clean[:, i:i+1, :, :] - self.scaled_epsilons[i], min=self.lower_bounds[i], max=self.upper_bounds[i])
            )
            upper_bounds.append(
                torch.clamp(image_clean[:, i:i+1, :, :] + self.scaled_epsilons[i], min=self.lower_bounds[i], max=self.upper_bounds[i])
            )
        lower_bound = torch.cat(lower_bounds, dim=1)
        upper_bound = torch.cat(upper_bounds, dim=1)

        init_start = torch.cat([torch.empty_like(image_clean[:, i:i+1, :, :]).uniform_(-self.scaled_epsilons[i], self.scaled_epsilons[i]) for i in range(3)], dim=1)
        adv = image_clean + init_start
        for _ in range(self.num_iter):
            adv.requires_grad = True
            logits = model(adv)                     # update it to get indices
            losses = criterion(logits.to(torch.float32), target_label)

            if self.grad_scaler is not None:
                scaled_g = torch.autograd.grad(self.grad_scaler.scale(losses), adv,
                                        retain_graph=False, create_graph=False)[0]
                g = scaled_g / self.grad_scaler.get_scale()
            else:
                g = torch.autograd.grad(losses, adv,
                                        retain_graph=True, create_graph=False)[0]

            adv = torch.cat([adv[:, i:i+1, :, :] + torch.sign(g[:, i:i+1, :, :]) * self.scaled_step_sizes[i] for i in range(3)], dim=1)


            adv = torch.where(adv > lower_bound, adv, lower_bound)
            adv = torch.where(adv < upper_bound, adv, upper_bound).detach()

        return adv, target_label

