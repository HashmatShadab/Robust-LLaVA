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


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, load_encoder=None, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        # if kwargs['torch_dtype']  is not already set, set it to torch.float16
        if 'torch_dtype' not in kwargs:
            kwargs['torch_dtype'] = torch.float16
        #kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)

            # mm_projector_weights = torch.load(os.path.join(model_path, 'dino_mm_projector.bin'), map_location='cpu')
            # mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            # model.load_state_dict(mm_projector_weights, strict=False)

            # mm_projector_weights = torch.load(os.path.join(model_path, 'advxl_giant_mm_projector.bin'), map_location='cpu')
            # mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            # model.load_state_dict(mm_projector_weights, strict=False)


        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))



        # check if model has attribute 'use_clip_encoder', 'use_dino_encoder', 'use_advxl_giant_encoder', 'use_advxl_huge_encoder'
        if hasattr(model.config, 'use_clip_encoder') and model.config.use_clip_encoder:
            vision_tower = model.get_vision_tower(model_name='clip')
            if not vision_tower.is_loaded:
                #vision_tower.load_model(device_map=device_map)
                if load_encoder is not None:
                    vision_tower.load_model(device_map=device_map, load_encoder=load_encoder, device=device)
                else:
                    vision_tower.load_model(device_map=device_map)
            if load_encoder is not None:
                vision_tower.to(device="cuda", dtype=kwargs['torch_dtype'])
            else:
                if device_map != 'auto':
                    vision_tower.to(device=device_map, dtype=kwargs['torch_dtype'])
            image_processor = vision_tower.image_processor

        if hasattr(model.config, 'use_dino_encoder') and model.config.use_dino_encoder:
            dino_tower = model.get_vision_tower(model_name='dino')
            if not dino_tower.is_loaded:
                dino_tower.load_model(device_map=device_map)
            # if device_map != 'auto':
            #     dino_tower.to(device=device_map, dtype=kwargs['torch_dtype'])
            dino_tower.to(device='cuda', dtype=kwargs['torch_dtype'])
            image_processor = dino_tower.image_processor

        if hasattr(model.config, 'use_advxl_giant_encoder') and model.config.use_advxl_giant_encoder:
            advxl_giant_tower = model.get_vision_tower(model_name='advxl_giant')
            if not advxl_giant_tower.is_loaded:
                advxl_giant_tower.load_model(device_map=device_map, model_name='advxl_giant')
            # if device_map != 'auto':
            #     advxl_giant_tower.to(device=device_map, dtype=kwargs['torch_dtype'])
            advxl_giant_tower.to(device='cuda', dtype=kwargs['torch_dtype'])
            image_processor = advxl_giant_tower.image_processor

        if hasattr(model.config, 'use_advxl_huge_encoder') and model.config.use_advxl_huge_encoder:
            advxl_huge_tower = model.get_vision_tower(model_name='advxl_huge')
            if not advxl_huge_tower.is_loaded:
                advxl_huge_tower.load_model(device_map=device_map, model_name='advxl_huge')
            # if device_map != 'auto':
            #     advxl_huge_tower.to(device=device_map, dtype=kwargs['torch_dtype'])
            advxl_huge_tower.to(device='cuda', dtype=kwargs['torch_dtype'])
            image_processor = advxl_huge_tower.image_processor

        # use_ares_vitl_21k_encoder
        if hasattr(model.config, 'use_ares_vitl_21k_encoder') and model.config.use_ares_vitl_21k_encoder:
            ares_vitl_21k_tower = model.get_vision_tower(model_name='ares_vitl_21k')
            if not ares_vitl_21k_tower.is_loaded:
                ares_vitl_21k_tower.load_model(device_map=device_map, model_name='vitl_21k')
            # if device_map != 'auto':
            #     ares_vitl_21k_tower.to(device=device_map, dtype=torch.float16)
            ares_vitl_21k_tower.to(device='cuda', dtype=torch.float16)
            image_processor = ares_vitl_21k_tower.image_processor
        # use_ares_vitb_at_encoder
        if hasattr(model.config, 'use_ares_vitb_at_encoder') and model.config.use_ares_vitb_at_encoder:
            ares_vitb_at_tower = model.get_vision_tower(model_name='ares_vitb_at')
            if not ares_vitb_at_tower.is_loaded:
                ares_vitb_at_tower.load_model(device_map=device_map, model_name='vitb_at')
            # if device_map != 'auto':
            #     ares_vitb_at_tower.to(device=device_map, dtype=torch.float16)
            ares_vitb_at_tower.to(device='cuda', dtype=torch.float16)
            image_processor = ares_vitb_at_tower.image_processor

        # check if model does not have attribute 'use_clip_encoder', 'use_dino_encoder', 'use_advxl_giant_encoder', 'use_advxl_huge_encoder'
        if not hasattr(model.config, 'use_clip_encoder') and not hasattr(model.config, 'use_dino_encoder') \
            and not hasattr(model.config, 'use_advxl_giant_encoder') and not hasattr(model.config, 'use_advxl_huge_encoder') \
            and not hasattr(model.config, 'use_ares_vitl_21k_encoder') and not hasattr(model.config, 'use_ares_vitb_at_encoder'):
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                if load_encoder is not None:
                    vision_tower.load_model(device_map=device_map, load_encoder=load_encoder, device=device)
                else:
                    vision_tower.load_model(device_map=device_map)
            if load_encoder is not None:
                vision_tower.to(device="cuda", dtype=kwargs['torch_dtype'])
            else:
                if device_map != 'auto':
                    vision_tower.to(device=device_map, dtype=kwargs['torch_dtype'])
            image_processor = vision_tower.image_processor



    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
