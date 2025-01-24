""" A Wrapper for existing models in PyTorch

Hacked together by / Copyright 2024, Zeyu Wang
"""
import torch
import torch.nn as nn

from .attacker import NoOpAttacker, PGDAttacker


class AdvModel(nn.Module):
    def __init__(
            self, model, attacker=NoOpAttacker()):
        super().__init__()
        # in case a ddp is passed
        self.base_model = model
        self.default_cfg = model.default_cfg
        self.attacker = attacker


    def get_attacker(self):
        return self.attacker

    def set_attacker(self, attacker):
        assert attacker is not None
        self.attacker = attacker

    def forward(self, x, labels, criterion): # update it to get indices
        if isinstance(self.attacker, NoOpAttacker):
            images = x
        else:
            # need to enable gradient compute
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(True)

            if self.training:
                self.eval()
                aux_images, _ = self.attacker.attack(x, labels, self.base_model, criterion)
                self.train()
            else:
                aux_images, _ = self.attacker.attack(x, labels, self.base_model, criterion) # update it to get indices
            images = aux_images

            torch.set_grad_enabled(prev)

        return self.base_model(images) # update it to get indices