import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class DINOVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()


        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):

        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        ################# Why is clip vision tower constructed here ???? #############################
        # self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        # self.clip_vision_tower.requires_grad_(False)

        self.vision_tower = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        setattr(self.vision_tower, 'hidden_size', 1024)

        # should we use device map

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
                image_forward_out = self.vision_tower.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
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
        #return self.config.hidden_size
        return self.vision_tower.hidden_size

    @property
    def num_patches(self):
        #return (self.config.image_size // self.config.patch_size) ** 2
        return 256

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
