import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .dino_encoder import DINOVisionTower
from .advxl_encoder import AdvXLVisionTower
from .ares_encoder import ARESVisionTower


def build_vision_tower(vision_tower_cfg, load_model="clip", **kwargs):


    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            if load_model == "clip":
                return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            elif load_model == "dino":
                return DINOVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            elif load_model == "advxl_huge":
                return AdvXLVisionTower(vision_tower, args=vision_tower_cfg, model_name="advxl_huge", **kwargs)
            elif load_model == "advxl_giant":
                return AdvXLVisionTower(vision_tower, args=vision_tower_cfg, model_name="advxl_giant", **kwargs)
            elif load_model == "ares_vitl_21k":
                return ARESVisionTower(vision_tower, args=vision_tower_cfg, model_name="vitl_21k", **kwargs)
            elif load_model == "ares_vitb_at":
                return ARESVisionTower(vision_tower, args=vision_tower_cfg, model_name="vitb_at", **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {load_model}")



    raise ValueError(f'Unknown vision tower: {vision_tower}')
