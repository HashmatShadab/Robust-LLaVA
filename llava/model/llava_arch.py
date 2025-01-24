#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):

            # check if the config has attribute use_clip_encoder
            if getattr(config, "use_clip_encoder", False):
                self.vision_tower = build_vision_tower(config, load_model="clip", delay_load=True)
                config.mm_hidden_size = 1024
                self.mm_projector = build_vision_projector(config)

            # check if the config has attribute use_dino_encoder
            if getattr(config, "use_dino_encoder", False):
                self.dino_tower = build_vision_tower(config, load_model="dino", delay_load=True)
                self.dino_mm_projector = build_vision_projector(config)

            # check if the config has attribute use_advxl_giant_encoder
            if getattr(config, "use_advxl_giant_encoder", False):
                self.advxl_giant_tower = build_vision_tower(config, load_model="advxl_giant", delay_load=True)
                config.mm_hidden_size = 1408
                self.advxl_giant_mm_projector = build_vision_projector(config)

            # check if the config has attribute use_advxl_huge_encoder
            if getattr(config, "use_advxl_huge_encoder", False):
                self.advxl_huge_tower = build_vision_tower(config, load_model="advxl_huge", delay_load=True)
                config.mm_hidden_size = 1280
                self.advxl_huge_mm_projector = build_vision_projector(config)

            if getattr(config, "use_ares_vitl_21k_encoder", False):
                self.ares_vitl_21k_tower = build_vision_tower(config, load_model="ares_vitl_21k", delay_load=True)
                config.mm_hidden_size = 1024
                self.ares_vitl_21k_mm_projector = build_vision_projector(config)

            if getattr(config, "use_ares_vitb_at_encoder", False):
                self.ares_vitb_at_tower = build_vision_tower(config, load_model="ares_vitb_at", delay_load=True)
                config.mm_hidden_size = 768
                self.ares_vitb_at_mm_projector = build_vision_projector(config)

            # if all the above conditions are not met, load clip encoder
            if not getattr(config, "use_clip_encoder", False) and not getattr(config, "use_dino_encoder", False) \
                and not getattr(config, "use_advxl_giant_encoder", False) and not getattr(config, "use_advxl_huge_encoder", False) \
                and not getattr(config, "use_ares_vitl_21k_encoder", False) and not getattr(config, "use_ares_vitb_at_encoder", False):
                self.vision_tower = build_vision_tower(config, load_model="clip", delay_load=True)
                self.mm_projector = build_vision_projector(config)



            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self, load_model="clip"):

        if load_model == "clip":
            vision_tower = getattr(self, 'vision_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower
        elif load_model == "dino":
            vision_tower = getattr(self, 'dino_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower
        elif load_model == "advxl_giant":
            vision_tower = getattr(self, 'advxl_giant_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower
        elif load_model == "advxl_huge":
            vision_tower = getattr(self, 'advxl_huge_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower
        elif load_model == "ares_vitl_21k":
            vision_tower = getattr(self, 'ares_vitl_21k_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower
        elif load_model == "ares_vitb_at":
            vision_tower = getattr(self, 'ares_vitb_at_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower

        else:
            raise NotImplementedError

    def initialize_vision_modules(self, model_args, fsdp=None):
        """
        Initializes the vision-related modules and configurations for the LlavaMetaModel.

        Configurations set:
            config.use_mm_proj: Enables the use of a multi-modal projector.
            config.mm_projector_type: Sets the projector type for multi-modal tasks,
                                      extracted from model_args. Defaults to 'linear'.
            config.mm_hidden_size: Sets the hidden size for the vision model if vision_tower is provided.
            config.mm_vision_select_layer: Sets the specific layer to be used from the vision model.
            config.mm_vision_select_feature: Sets the specific feature(patch, class) to be used from the vision model.
            config.mm_patch_merge_type: Sets the type of patch merge operation in the vision model(flat).
        """

        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        pretrain_dino_mm_mlp_adapter = model_args.pretrain_dino_mm_mlp_adapter
        pretrain_advxl_giant_mm_mlp_adapter = model_args.pretrain_advxl_giant_mm_mlp_adapter
        pretrain_advxl_huge_mm_mlp_adapter = model_args.pretrain_advxl_huge_mm_mlp_adapter
        pretrain_ares_vitl_21k_mm_mlp_adapter = model_args.pretrain_ares_vitl_21k_mm_mlp_adapter
        pretrain_ares_vitb_at_mm_mlp_adapter = model_args.pretrain_ares_vitb_at_mm_mlp_adapter


        self.config.mm_vision_tower = vision_tower


        if self.get_vision_tower() is None and model_args.use_clip_encoder:
            vision_tower = build_vision_tower(model_args, load_model="clip")

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        elif model_args.use_clip_encoder:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        if self.get_vision_tower(load_model="advxl_giant") is None and model_args.use_advxl_giant_encoder:
            advxl_giant_tower = build_vision_tower(model_args, load_model="advxl_giant")

            if fsdp is not None and len(fsdp) > 0:
                self.advxl_giant_tower = [advxl_giant_tower]
            else:
                self.advxl_giant_tower = advxl_giant_tower
        elif model_args.use_advxl_giant_encoder:
            if fsdp is not None and len(fsdp) > 0:
                advxl_giant_tower = self.advxl_giant_tower[0]
            else:
                advxl_giant_tower = self.advxl_giant_tower
            advxl_giant_tower.load_model()

        if self.get_vision_tower(load_model="advxl_huge") is None and model_args.use_advxl_huge_encoder:
            advxl_huge_tower = build_vision_tower(model_args, load_model="advxl_huge")

            if fsdp is not None and len(fsdp) > 0:
                self.advxl_huge_tower = [advxl_huge_tower]
            else:
                self.advxl_huge_tower = advxl_huge_tower
        elif model_args.use_advxl_huge_encoder:
            if fsdp is not None and len(fsdp) > 0:
                advxl_huge_tower = self.advxl_huge_tower[0]
            else:
                advxl_huge_tower = self.advxl_huge_tower
            advxl_huge_tower.load_model()

        if self.get_vision_tower(load_model="dino") is None and model_args.use_dino_encoder:
            dino_tower = build_vision_tower(model_args, load_model="dino")

            if fsdp is not None and len(fsdp) > 0:
                self.dino_tower = [dino_tower]
            else:
                self.dino_tower = dino_tower
        elif model_args.use_dino_encoder:
            if fsdp is not None and len(fsdp) > 0:
                dino_tower = self.dino_tower[0]
            else:
                dino_tower = self.dino_tower
            dino_tower.load_model()

        if self.get_vision_tower(load_model="ares_vitl_21k") is None and model_args.use_ares_vitl_21k_encoder:
            ares_vitl_21k_tower = build_vision_tower(model_args, load_model="ares_vitl_21k")

            if fsdp is not None and len(fsdp) > 0:
                self.ares_vitl_21k_tower = [ares_vitl_21k_tower]
            else:
                self.ares_vitl_21k_tower = ares_vitl_21k_tower
        elif model_args.use_ares_vitl_21k_encoder:
            if fsdp is not None and len(fsdp) > 0:
                ares_vitl_21k_tower = self.ares_vitl_21k_tower[0]
            else:
                ares_vitl_21k_tower = self.ares_vitl_21k_tower
            ares_vitl_21k_tower.load_model()

        if self.get_vision_tower(load_model="ares_vitb_at") is None and model_args.use_ares_vitb_at_encoder:
            ares_vitb_at_tower = build_vision_tower(model_args, load_model="ares_vitb_at")

            if fsdp is not None and len(fsdp) > 0:
                self.ares_vitb_at_tower = [ares_vitb_at_tower]
            else:
                self.ares_vitb_at_tower = ares_vitb_at_tower
        elif model_args.use_ares_vitb_at_encoder:
            if fsdp is not None and len(fsdp) > 0:
                ares_vitb_at_tower = self.ares_vitb_at_tower[0]
            else:
                ares_vitb_at_tower = self.ares_vitb_at_tower
            ares_vitb_at_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        # self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.use_clip_encoder = model_args.use_clip_encoder
        self.config.use_advxl_giant_encoder = model_args.use_advxl_giant_encoder
        self.config.use_advxl_huge_encoder = model_args.use_advxl_huge_encoder
        self.config.use_dino_encoder = model_args.use_dino_encoder
        self.config.use_ares_vitl_21k_encoder = model_args.use_ares_vitl_21k_encoder
        self.config.use_ares_vitb_at_encoder = model_args.use_ares_vitb_at_encoder

        if getattr(self, 'mm_projector', None) is None and model_args.use_clip_encoder:
            self.config.mm_hidden_size = vision_tower.hidden_size
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        elif model_args.use_clip_encoder:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'advxl_giant_mm_projector', None) is None and model_args.use_advxl_giant_encoder:
            self.config.mm_hidden_size = advxl_giant_tower.hidden_size
            self.advxl_giant_mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        elif model_args.use_advxl_giant_encoder:
            # In case it is frozen by LoRA
            for p in self.advxl_giant_mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'advxl_huge_mm_projector', None) is None and model_args.use_advxl_huge_encoder:
            self.config.mm_hidden_size = advxl_huge_tower.hidden_size
            self.advxl_huge_mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        elif model_args.use_advxl_huge_encoder:
            # In case it is frozen by LoRA
            for p in self.advxl_huge_mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'dino_mm_projector', None) is None and model_args.use_dino_encoder:
            self.config.mm_hidden_size = dino_tower.hidden_size
            self.dino_mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        elif model_args.use_dino_encoder:
            # In case it is frozen by LoRA
            for p in self.dino_mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'ares_vitl_21k_mm_projector', None) is None and model_args.use_ares_vitl_21k_encoder:
            self.config.mm_hidden_size = ares_vitl_21k_tower.hidden_size
            self.ares_vitl_21k_mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        elif model_args.use_ares_vitl_21k_encoder:
            # In case it is frozen by LoRA
            for p in self.ares_vitl_21k_mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, 'ares_vitb_at_mm_projector', None) is None and model_args.use_ares_vitb_at_encoder:
            self.config.mm_hidden_size = ares_vitb_at_tower.hidden_size
            self.ares_vitb_at_mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        elif model_args.use_ares_vitb_at_encoder:
            # In case it is frozen by LoRA
            for p in self.ares_vitb_at_mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None and model_args.use_clip_encoder:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword)[1]: v for k, v in weights.items() if keyword in k}

            msg = self.mm_projector.load_state_dict(get_w(mm_projector_weights, '.mm_projector.'))

        if pretrain_advxl_giant_mm_mlp_adapter is not None and model_args.use_advxl_giant_encoder:
            # print("Loading pretrained DINO adpater!!!")
            advxl_giant_mm_projector_weights = torch.load(pretrain_advxl_giant_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword)[1]: v for k, v in weights.items() if keyword in k}

            msg = self.advxl_giant_mm_projector.load_state_dict(get_w(advxl_giant_mm_projector_weights, '.advxl_giant_mm_projector.'))

        if pretrain_advxl_huge_mm_mlp_adapter is not None and model_args.use_advxl_huge_encoder:
            # print("Loading pretrained DINO adpater!!!")
            advxl_huge_mm_projector_weights = torch.load(pretrain_advxl_huge_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword)[1]: v for k, v in weights.items() if keyword in k}

            msg = self.advxl_huge_mm_projector.load_state_dict(get_w(advxl_huge_mm_projector_weights, '.advxl_huge_mm_projector.'))

        if pretrain_dino_mm_mlp_adapter is not None and model_args.use_dino_encoder:
            # print("Loading pretrained DINO adpater!!!")
            dino_mm_projector_weights = torch.load(pretrain_dino_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword)[1]: v for k, v in weights.items() if keyword in k}

            msg = self.dino_mm_projector.load_state_dict(get_w(dino_mm_projector_weights, '.dino_mm_projector.'))

        if pretrain_ares_vitl_21k_mm_mlp_adapter is not None and model_args.use_ares_vitl_21k_encoder:
            # print("Loading pretrained DINO adpater!!!")
            ares_vitl_21k_mm_projector_weights = torch.load(pretrain_ares_vitl_21k_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword)[1]: v for k, v in weights.items() if keyword in k}

            msg = self.ares_vitl_21k_mm_projector.load_state_dict(
                get_w(ares_vitl_21k_mm_projector_weights, '.ares_vitl_21k_mm_projector.'))

        if pretrain_ares_vitb_at_mm_mlp_adapter is not None and model_args.use_ares_vitb_at_encoder:
            # print("Loading pretrained DINO adpater!!!")
            ares_vitb_at_mm_projector_weights = torch.load(pretrain_ares_vitb_at_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword)[1]: v for k, v in weights.items() if keyword in k}

            msg = self.ares_vitb_at_mm_projector.load_state_dict(
                get_w(ares_vitb_at_mm_projector_weights, '.ares_vitb_at_mm_projector.'))
            # print(f"Loaded pretrained ares_vitb_at_mm_mlp_adapter from {pretrain_ares_vitb_at_mm_mlp_adapter} with message: {msg}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self, model_name="clip"):
        if model_name == "clip":
            return self.get_model().get_vision_tower(load_model='clip')
        elif model_name == "dino":
            return self.get_model().get_vision_tower(load_model="dino")
        elif model_name == "advxl_giant":
            return self.get_model().get_vision_tower(load_model="advxl_giant")
        elif model_name == "advxl_huge":
            return self.get_model().get_vision_tower(load_model="advxl_huge")
        elif model_name == "ares_vitl_21k":
            return self.get_model().get_vision_tower(load_model="ares_vitl_21k")
        elif model_name == "ares_vitb_at":
            return self.get_model().get_vision_tower(load_model="ares_vitb_at")
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def get_advxl_giant_vision_tower(self):
        return self.get_model().get_vision_tower(load_model='advxl_giant')

    def get_advxl_huge_vision_tower(self):
        return self.get_model().get_vision_tower(load_model='advxl_huge')

    def get_dino_vision_tower(self):
        return self.get_model().get_vision_tower(load_model='dino')

    def get_ares_vitl_21k_vision_tower(self):
        return self.get_model().get_vision_tower(load_model='ares_vitl_21k')

    def get_ares_vitb_at_vision_tower(self):
        return self.get_model().get_vision_tower(load_model='ares_vitb_at')

    def encode_images_withclip(self, images):
        image_features = self.get_model().get_vision_tower(load_model="clip")(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_withdino(self, images):
        image_features = self.get_model().get_vision_tower(load_model="dino")(images)
        image_features = self.get_model().dino_mm_projector(image_features)
        return image_features

    def encode_images_withadvxl_giant(self, images):
        image_features = self.get_model().get_vision_tower(load_model="advxl_giant")(images)
        image_features = self.get_model().advxl_giant_mm_projector(image_features)
        return image_features

    def encode_images_withadvxl_huge(self, images):
        image_features = self.get_model().get_vision_tower(load_model="advxl_huge")(images)
        image_features = self.get_model().advxl_huge_mm_projector(image_features)
        return image_features

    def encode_images_withares_vitl_21k(self, images):
        image_features = self.get_model().get_vision_tower(load_model="ares_vitl_21k")(images)
        image_features = self.get_model().ares_vitl_21k_mm_projector(image_features)
        return image_features

    def encode_images_withares_vitb_at(self, images):
        image_features = self.get_model().get_vision_tower(load_model="ares_vitb_at")(images)
        image_features = self.get_model().ares_vitb_at_mm_projector(image_features)
        return image_features

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal_withdino(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None
    ):

        vision_tower = self.get_vision_tower(model_name="clip") or self.get_vision_tower(model_name="dino") or self.get_vision_tower(model_name="advxl_giant") or self.get_vision_tower(model_name="advxl_huge") or self.get_vision_tower(model_name="ares_vitl_21k") or self.get_vision_tower(model_name="ares_vitb_at")

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            #print("I only used multiple images!")
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

            if mm_patch_merge_type == 'flat':
                image_features_1 = [x.flatten(0, 1) for x in image_features]
                image_features_2 = None

            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features_1 = new_image_features
                image_features_2 = None
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

        else:
            if getattr(self.config, 'use_clip_encoder', False) and getattr(self.config, 'use_dino_encoder', False):
                # Get image features from both CLIP and DINO encoders
                image_features_1 = self.encode_images_withclip(images)
                image_features_2 = self.encode_images_withdino(images)
            elif getattr(self.config, 'use_clip_encoder', False) and getattr(self.config, 'use_advxl_giant_encoder', False):
                # Get image features from both CLIP and AdvXL Giant encoders
                image_features_1 = self.encode_images_withclip(images)
                image_features_2 = self.encode_images_withadvxl_giant(images)
            elif getattr(self.config, 'use_clip_encoder', False) and getattr(self.config, 'use_advxl_huge_encoder', False):
                # Get image features from both CLIP and AdvXL Huge encoders
                image_features_1 = self.encode_images_withclip(images)
                image_features_2 = self.encode_images_withadvxl_huge(images)
            elif getattr(self.config, 'use_advxl_giant_encoder', False) and getattr(self.config, 'use_advxl_huge_encoder', False):
                # Get image features from both AdvXL Giant and AdvXL Huge encoders
                image_features_1 = self.encode_images_withadvxl_giant(images)
                image_features_2 = self.encode_images_withadvxl_huge(images)
            elif getattr(self.config, 'use_clip_encoder', False) and getattr(self.config, 'use_ares_vitl_21k_encoder',
                                                                             False):  # self.config.use_ares_vitl_21k_encoder:
                # Get image features from both CLIP and Ares VitL 21k encoders
                image_features_1 = self.encode_images_withclip(images)
                image_features_2 = self.encode_images_withares_vitl_21k(images)
            elif getattr(self.config, 'use_clip_encoder', False) and getattr(self.config, 'use_ares_vitb_at_encoder',
                                                                             False):  # self.config.use_ares_vitb_at_encoder:
                # Get image features from both CLIP and Ares VitB AT encoders
                image_features_1 = self.encode_images_withclip(images)
                image_features_2 = self.encode_images_withares_vitb_at(images)
            elif getattr(self.config, 'use_dino_encoder', False) and getattr(self.config, 'use_advxl_giant_encoder', False):
                # Get image features from both DINO and AdvXL Giant encoders
                image_features_1 = self.encode_images_withadvxl_giant(images)
                image_features_2 = self.encode_images_withdino(images)

            elif getattr(self.config, 'use_dino_encoder', False) and getattr(self.config, 'use_advxl_huge_encoder', False):
                # Get image features from both DINO and AdvXL Huge encoders
                image_features_1 = self.encode_images_withadvxl_huge(images)
                image_features_2 = self.encode_images_withdino(images)


            elif getattr(self.config, 'use_clip_encoder', False):
                image_features_1 = self.encode_images_withclip(images)
                image_features_2 = None
            elif getattr(self.config, 'use_dino_encoder', False):
                image_features_1 = self.encode_images_withdino(images)
                image_features_2 = None
            elif getattr(self.config, 'use_advxl_giant_encoder', False):
                image_features_1 = self.encode_images_withadvxl_giant(images)
                image_features_2 = None
            elif getattr(self.config, 'use_advxl_huge_encoder', False):
                image_features_1 = self.encode_images_withadvxl_huge(images)
                image_features_2 = None
            elif getattr(self.config, 'use_ares_vitl_21k_encoder', False):
                # Get image features from Ares VitL 21k encoder
                image_features_1 = self.encode_images_withares_vitl_21k(images)
                image_features_2 = None
            elif getattr(self.config, 'use_ares_vitb_at_encoder', False):
                # Get image features from Ares VitB AT encoder
                image_features_1 = self.encode_images_withares_vitb_at(images)
                image_features_2 = None
            elif not getattr(self.config, 'use_clip_encoder', False) and vision_tower:
                image_features_1 = self.encode_images_withclip(images)
                image_features_2 = None
            else:
                raise ValueError("No encoder is selected!")

            # image_features_1 = self.encode_images_withclip(images)
            # image_features_2 = self.encode_images_withadvxl_giant(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                          False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features_1[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []


            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:

                    cur_image_features_1 = image_features_1[cur_image_idx]
                    cur_image_features_2 = None if image_features_2 is None else image_features_2[cur_image_idx]
                    cur_image_idx += 1

                    if cur_image_features_2 is not None:
                        num_patches, clip_dim = cur_image_features_1.shape
                        clip_dtype = cur_image_features_1.dtype

                        # Interleave features
                        merged_features = torch.empty(2 * num_patches, clip_dim, dtype=clip_dtype)
                        merged_features[0::2] = cur_image_features_1
                        merged_features[1::2] = cur_image_features_2
                        cur_new_input_embeds.append(merged_features)

                        cur_new_labels.append(torch.full((merged_features.shape[0],),
                                                         IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    else:
                        cur_new_input_embeds.append(cur_image_features_1)
                        cur_new_labels.append(torch.full((cur_image_features_1.shape[0],),
                                                         IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))


            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype,
                                     device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        """
        Prepares inputs, labels, and attention masks for a multimodal model. The image is passed through the vision tower and the resulting features are merged with the input.

        Parameters:
        - input_ids: A tensor of shape (batch_size, sequence_length) containing input IDs.
        - attention_mask: A tensor of shape (batch_size, sequence_length) containing attention mask values.
        - labels: A tensor of shape (batch_size, sequence_length) containing labels for each input token.
        - images: Either a list of tensors or a single tensor representing the input images.
        - image_sizes: Optional image sizes if the images are not all the same size.


        This method prepares the inputs, labels, and attention masks for a multimodal model. It handles cases where images are present in the input and processes them accordingly.
        If images are not present or if the input is empty, the method simply returns the original inputs, labels, and attention masks.

        If images are present, the method encodes the images using the `encode_images` method and performs the necessary operations to merge the image features with the input.
        The `mm_patch_merge_type` and `image_aspect_ratio` parameters determine how the image features are merged. The resulting image features and labels are returned.

        The method also handles cases where attention masks, position IDs, and labels are not provided. In such cases, default values are used to ensure compatibility with the model.

        Finally, the method truncates the sequences to the maximum length defined by the `tokenizer_model_max_length` parameter and pads the input and label tensors to match the maximum length.


        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        """
            Initializes the tokenizer for vision-related tasks by adding special tokens and resizing the token embeddings accordingly.

            Args:
                model_args: Object containing model arguments. It should have the following attributes:
                    - mm_use_im_patch_token: Boolean flag to indicate if an image patch token should be used.
                    - mm_use_im_start_end: Boolean flag to indicate if start and end tokens for images should be used.
                    - tune_mm_mlp_adapter: Boolean flag to indicate if the MLP adapter should be tunable.
                    - pretrain_mm_mlp_adapter: Path to a pretrained MLP adapter model.
                tokenizer: Tokenizer object to be initialized.

            Tokenizer Initialization:
                - Adds default image patch token if `mm_use_im_patch_token` is set.
                - Adds default start and end tokens for images if `mm_use_im_start_end` is set.
                - Resizes the token embeddings after adding new tokens.

            Embedding Adjustments:
                - Adjusts the input and output embeddings weights for the new tokens by taking the average of existing embeddings.
                - Sets `requires_grad` flag based on tuning and pretraining flags for MLP adapter.

            Raises:
                ValueError: If the shape of pretrained `embed_tokens_weight` does not match the expected shape and cannot be used
                            directly for new tokens.
            """
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
