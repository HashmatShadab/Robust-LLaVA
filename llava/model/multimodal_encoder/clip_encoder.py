import torch
import torch.nn as nn
import os
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class ClipVisionModel(torch.nn.Module):
    def __init__(self, model,  normalize, all_tokens=False, proj=True):
        super().__init__()
        self.model = model
        self.normalize = normalize
        self.proj = model.proj
        if all_tokens:
            self.model.output_tokens = True
        if not proj:
            self.model.proj = None

    def forward(self, vision_, output_normalize=False):
        embedding = self.model(self.normalize(vision_))
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)

        if self.model.output_tokens:
            # flatten and concatenate all tokens
            return torch.hstack([embedding[0].flatten(1), embedding[1].flatten(1)])
        else:
            return embedding


class CLIPVisionTower(nn.Module):
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

    def load_model(self, device_map=None, device="cuda", load_encoder=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.load_encoder = load_encoder

        if self.load_encoder in ["simclip2", "simclip4", "fare2", "fare4", "clip224"]:
            import open_clip
            print("using open_clip")
            model_orig, _, image_processor = open_clip.create_model_and_transforms('ViT-L-14')
            vision_model = model_orig.visual

            if load_encoder == "simclip2":
                ckpt_path = "/mnt/nvme0n1/Dataset/muzammal/robust_llava_weights/baseline_models/simclip2.pt"

            elif load_encoder == "simclip4":
                ckpt_path = "/mnt/nvme0n1/Dataset/muzammal/robust_llava_weights/baseline_models/simclip4.pt"

            elif load_encoder == "fare2":
                ckpt_path = "/mnt/nvme0n1/Dataset/muzammal/robust_llava_weights/baseline_models/fare_eps_2.pt"

            elif load_encoder == "fare4":
                ckpt_path = "/mnt/nvme0n1/Dataset/muzammal/robust_llava_weights/baseline_models/fare_eps_4.pt"


            if self.load_encoder == "clip224":
                print("loading clip224")
                ckpt_path = "pretrained"
            else:
                vision_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            self.image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')  # 224
            # llava operates on the second to last layer output, so we remove the last layer
            vision_model.transformer.resblocks = vision_model.transformer.resblocks[:-1];
            print(f"Loaded CLIP Encoder {load_encoder} with ckpt {ckpt_path}")
            print("removing last layer of vision model")
            model_orig = ClipVisionModel(
                model=vision_model,
                normalize=lambda x: x,  # images have to be normalized, e.g. as handled by the llava model wrapper
                all_tokens=True, proj=False
            )
            self.vision_tower = model_orig
            self.vision_tower.device = device

        elif self.load_encoder in ["fare4_336"]:
            import open_clip
            print("using open_clip")
            model_orig, _, image_processor = open_clip.create_model_and_transforms('ViT-L-14-336')
            vision_model = model_orig.visual


            if load_encoder == "fare4_336":
                ckpt_path = "/path/to/baseline_models/fare_eps_4_336.pt"


            vision_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            self.image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')  # 336 openai/clip-vit-large-patch14-336
            # llava operates on the second to last layer output, so we remove the last layer
            vision_model.transformer.resblocks = vision_model.transformer.resblocks[:-1];
            print(f"Loaded CLIP Encoder {load_encoder} with ckpt {ckpt_path}")
            print("removing last layer of vision model")
            model_orig = ClipVisionModel(
                model=vision_model,
                normalize=lambda x: x,  # images have to be normalized, e.g. as handled by the llava model wrapper
                all_tokens=True, proj=False
            )
            self.vision_tower = model_orig
            self.vision_tower.device = device

        else:

            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

            print("Loaded CLIP Encoder")

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if self.load_encoder in ["simclip2", "simclip4", "fare2", "fare4", "clip224", "fare4_336"]:
            image_features = image_forward_outs
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

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
                if self.load_encoder in ["simclip2", "simclip4", "fare2", "fare4", "clip224",  "fare4_336"]:
                    image_forward_out = self.vision_tower(image.to(device=self.device).unsqueeze(0)).reshape(images.shape[0], -1, 1024)
                else:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            if self.load_encoder in ["simclip2", "simclip4", "fare2", "fare4", "clip224", "fare4_336"]:
                image_forward_outs = self.vision_tower(images.to(device=self.device)).reshape(images.shape[0], -1, 1024)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
