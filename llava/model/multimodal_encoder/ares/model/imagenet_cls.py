import torch
import os
import gdown
import torch.nn as nn
from timm.models import create_model
from .resnet import resnet50, wide_resnet50_2, ResNetGELU
from .imagenet_model_zoo import imagenet_model_zoo
from ..utils.registry import registry


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # Here we assume the color channel is in at dim=1

    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def denormalize(tensor, mean, std):
    '''
    Args:
        tensor (torch.Tensor): Float tensor image of size (B, C, H, W) to be denormalized.
        mean (torch.Tensor): float tensor means of size (C, )  for each channel.
        std (torch.Tensor): float tensor standard deviations of size (C, ) for each channel.
    '''
    return tensor*std[None]+mean[None]

class NormalizeByChannelMeanStd(nn.Module):
    '''The class of a normalization layer.'''
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

@registry.register_model('ImageNetCLS')
class ImageNetCLS(torch.nn.Module):
    '''The class to create ImageNet model.'''
    def __init__(self, model_name, normalize=True):
        '''
        Args:
            model_name (str): The model name in the ImageNet model zoo.
            normalize (bool): Whether interating the normalization layer into the model.
        '''
        super().__init__()
        self.model_name = model_name
        self.normalize = normalize
        self.backbone = imagenet_model_zoo[self.model_name]['model']
        mean=imagenet_model_zoo[self.model_name]['mean']
        std=imagenet_model_zoo[self.model_name]['std']
        self.pretrained=imagenet_model_zoo[self.model_name]['pretrained']
        act_gelu=imagenet_model_zoo[self.model_name]['act_gelu']

        if self.backbone=='resnet50_rl':
            model=resnet50()
        elif self.backbone=='wide_resnet50_2_rl':
            model=wide_resnet50_2()
        else:
            model_kwargs=dict({'num_classes': 0})
            if act_gelu:
                model_kwargs['act_layer']=ResNetGELU
            model = create_model(self.backbone, pretrained=self.pretrained, **model_kwargs)
        self.model=model
        self.model.norm = torch.nn.Identity()

        self.url = imagenet_model_zoo[self.model_name]['url']

        ckpt_name = '' if self.pretrained else imagenet_model_zoo[self.model_name]['pt']
        self.model_path=os.path.join(registry.get_path('cache_dir'), ckpt_name)

        if self.url:
            gdown.download(self.url, self.model_path, quiet=False, resume=True)

        self.load()

        if self.normalize:
            normalization = NormalizeByChannelMeanStd(mean=mean, std=std)
            self.model = torch.nn.Sequential(normalization, self.model)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): The input images. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].

        Returns:
            torch.Tensor: The output logits with shape [N D].

        '''

        labels = self.model.forward_features(x)
        return labels

    def load(self):
        '''The function to load ckpt.'''
        if not self.pretrained:
            ckpt=torch.load(self.model_path, map_location='cpu')
            msg=self.model.load_state_dict(ckpt, strict=False)
            print(f'Load ckpt from {self.model_path}. with msg: {msg}')


