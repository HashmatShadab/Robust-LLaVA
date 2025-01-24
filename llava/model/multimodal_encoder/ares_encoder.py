import torch
import torch.nn as nn
import os
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from .ares.utils.registry import registry
from .ares.model import  imagenet_model_zoo


# if __name__ == "__main__":
#     model_cls = registry.get_model('ImageNetCLS')
    # model = model_cls("swinl_at", normalize=False)

    # test forward pass
    # x = torch.randn(1, 3, 224, 224)
    # out = model(x)
    # print(out.shape)
    #
    # model = model_cls("swinl_21k", normalize=False)
    #
    # # test forward pass
    # x = torch.randn(1, 3, 224, 224)
    # out = model(x)
    # print(out.shape)

    # model = model_cls("vitl_21k", normalize=False)
    #
    # # test forward pass
    # x = torch.randn(1, 3, 224, 224)
    # out = model(x)
    # print(out.shape)
    #
    # model = model_cls("vitb_at", normalize=False)
    #
    # # test forward pass
    # x = torch.randn(1, 3, 224, 224)
    # out = model(x)
    # print(out.shape)

class ARESVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, model_name="vitl_21k"):
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

        # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        self.image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')  # 224

        model_cls = registry.get_model('ImageNetCLS')
        self.vision_tower = model_cls(model_name, normalize=False)
        if model_name == "vitl_21k":
            setattr(self.vision_tower, 'hidden_size', 1024)
        elif model_name == "vitb_at":
            setattr(self.vision_tower, 'hidden_size', 768)


        self.vision_tower.requires_grad_(False)


        self.is_loaded = True
        #print(f"Loaded {model_name} for vision tower")

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
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
