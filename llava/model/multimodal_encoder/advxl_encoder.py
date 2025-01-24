import torch
import torch.nn as nn
import os
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


from .timm_xl.models import create_model as create_model_xl
from .timm_xl.models import load_checkpoint as load_checkpoint_xl

class AdvXLVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, model_name="advxl_giant"):
        super().__init__()


        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model(model_name)
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model(model_name)
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, model_name, device_map=None):

        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        # self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        # self.clip_vision_tower.requires_grad_(False)

        if model_name == "advxl_giant":
            model_name = "vit_giant_patch14_224"
            ckpt = "path/to/advxl_vit_g14.pth"

        elif model_name == "advxl_huge":
            model_name = "vit_huge_patch14_224"
            ckpt = "path/to/advxl/advxl_vit_h14.pth"
        else:
            raise ValueError(f'Unexpected model name: {model_name}')

        self.vision_tower = create_model_xl(model_name, pretrained=False, num_classes=0, in_chans=3,
                                global_pool='avg', scriptable=False, img_size=336)
        if model_name == "vit_giant_patch14_224":
            setattr(self.vision_tower, 'hidden_size', 1408)
        elif model_name == "vit_huge_patch14_224":
            setattr(self.vision_tower, 'hidden_size', 1280)
        else:
            raise ValueError(f'Unexpected model name: {model_name}')

        msg = load_checkpoint_xl(self.vision_tower, ckpt, strict=False)

        self.vision_tower.requires_grad_(False)


        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs["x_prenorm"]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    #@torch.no_grad() In order to generate adversarial examples
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower.forward_features_(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower.forward_features_(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        # if self.is_loaded:
        #     return self.clip_vision_tower.config
        # else:
        #     return self.cfg_only
        assert NotImplementedError
        pass

    @property
    def hidden_size(self):
        #return
        return self.vision_tower.hidden_size

    @property
    def num_patches(self):
        #return (self.config.image_size // self.config.patch_size) ** 2
        return 256

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
