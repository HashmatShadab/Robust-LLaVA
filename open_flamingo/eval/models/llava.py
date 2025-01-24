import copy
import os

from typing import List

import torch

from torchvision.transforms import transforms

from open_flamingo.eval.eval_model import BaseEvalModel
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle


class EvalModelLLAVA(BaseEvalModel):
    """LLaVA model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, model_args):
        super().__init__(model_args)
        disable_torch_init()
        model_path = os.path.expanduser(model_args["model_path"])
        model_name = get_model_name_from_path(model_path)
        if model_args['precision'] == 'float16':
            torch_dtype = torch.float16
        elif model_args['precision'] == 'float32':
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unknown dtype: {model_args['precision']}")
        # load_encoder is None if model_args does not have 'load_encoder'
        load_encoder = model_args.get('load_encoder', None)
        if load_encoder == "none" or load_encoder == "None":
            load_encoder = None
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            model_path, model_args.get("model_base"), model_name,
            torch_dtype=torch_dtype, load_encoder=load_encoder
            )
        # self.model.base_model.advxl_giant_mm_projector.state_dict().keys()
        self.image_processor.do_normalize = False
        self.normalizer = transforms.Normalize(
            mean=self.image_processor.image_mean, std=self.image_processor.image_std
            )  # we need to normalize in the forward pass, so that the threat model is consistent
        model_args["temperature"] = float(model_args["temperature"])
        model_args["num_beams"] = int(model_args["num_beams"])
        self.model_args = model_args
        self.conv_mode = "vicuna_v1"
        if model_args["precision"] == "float16":
            self.cast_dtype = torch.float16
        elif model_args["precision"] == "float32":
            self.cast_dtype = torch.float32
        else:
            raise ValueError(f"Unknown dtype: {model_args['precision']}")

        self.dataset_name = model_args.get("dataset_name")

        self.stop_str = conv_templates[self.conv_mode].sep if conv_templates[self.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[self.conv_mode].sep2
        self.stop_token_id = self.tokenizer.convert_tokens_to_ids(self.stop_str)

    @torch.no_grad()
    def get_outputs(
        self,
        batch_text,  # List[conv object]
        batch_images: torch.Tensor,
        min_generation_length: int,
        max_generation_length: int,
        **kwargs,
    ) -> List[str]:
        assert len(batch_text) == 1, "Only support batch size 1 (yet)"
        assert 0. <= batch_images.min() and batch_images.max() <= 1., "Images must be in image space"

        #prompt = batch_text.get_prompt()
        input_ids = self._prepare_text(batch_text)

        batch_images = self.normalizer(batch_images)
        output_ids = self.model.generate(
            input_ids,
            images=batch_images.to(dtype=self.cast_dtype, device='cuda', non_blocking=True),
            do_sample=True if self.model_args["temperature"] > 0 else False,
            temperature=self.model_args["temperature"],
            top_p=self.model_args.get("top_p"),
            num_beams=self.model_args["num_beams"],
            min_new_tokens=min_generation_length,
            max_new_tokens=max_generation_length,
            use_cache=False
        )

        input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        #outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()

        if outputs.endswith(self.stop_str):
            outputs = outputs[:-len(self.stop_str)]
        outputs = outputs.strip()

        return [outputs]

    def __call__(self, images_unnorm):
        assert self.input_ids is not None
        assert self.attention_mask is not None
        assert self.labels is not None
        assert 0. <= images_unnorm.min() and images_unnorm.max() <= 1., "Images must be in image space"
        assert len(images_unnorm.shape) == 4, "[b, c, h, w]"

        out = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            past_key_values=self.past_key_values,
            inputs_embeds=None,
            labels=self.labels,
            images=self.normalizer(images_unnorm),
        )
        return out.loss.unsqueeze(0)

    def set_inputs(
        self,
        batch_text,
        past_key_values: torch.Tensor = None,
        to_device: bool = False,
    ):
        self.input_ids = self._prepare_text(batch_text)

        context_only = batch_text[0].get_prompt().split("ASSISTANT:")[0] + "ASSISTANT:"
        context_len = len(self.tokenizer.encode(context_only))

        labels = copy.deepcopy(self.input_ids)
        labels[:, :context_len] = IGNORE_INDEX
        # labels[labels == self.stop_token_id] = IGNORE_INDEX
        # print(batch_text[0].get_prompt())
        # print(self.tokenizer.decode(labels[labels != IGNORE_INDEX]))
        self.labels = labels
        self.attention_mask = self.input_ids.ne(self.tokenizer.pad_token_id)
        self.past_key_values = past_key_values


    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        assert len(batch) == 1, "Only support batch size 1 (yet)"
        image_tensor = process_images(batch[0], self.image_processor, self.model.config)
        return image_tensor

    def _prepare_text(self, convs):
        input_ids = [
            tokenizer_image_token(conv.get_prompt(), self.tokenizer, return_tensors='pt') for conv in convs
        ]
        input_ids = torch.stack(input_ids, dim=0).to(device='cuda', non_blocking=True)
        return input_ids

    def get_vqa_prompt(self, question, answer=None) -> str:
        if self.dataset_name == "vizwiz":
            self.prompt_suffix = "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
        elif self.dataset_name == "textvqa":
            self.prompt_suffix = "\nAnswer the question using a single word or phrase."
        elif self.dataset_name == "ok_vqa":
            self.prompt_suffix = "\nAnswer the question using a single word or phrase."
        elif self.dataset_name == "vqav2":
            self.prompt_suffix = "\nAnswer the question using a single word or phrase."
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            self.prompt_suffix = ""
            print(f"Unknown dataset: {DATASET_NAME}, using no prompt suffix.")

        qs = question + self.prompt_suffix

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], answer)

        return conv

    def get_caption_prompt(self, caption=None) -> str:
        qs = "Provide a short caption for this image."

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], caption)

        return conv

    def get_caption_prompt_ap(self, caption=None) -> str:
        qs = "Given the image could be adversarially perturbed. Provide a short caption for this image."

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], caption)

        return conv

    def get_caption_prompt_ac(self, caption=None) -> str:
        qs = "Consider the given image being adversarially perturbed. Provide a short caption for this image."

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], caption)

        return conv

    def get_caption_prompt_ac_2(self, caption=None) -> str:
        qs = "Given the image is adversarially perturbed. Provide a short caption for this image."
        # Consider thegivenimagebeingadversariallyperturbed. Provide a short caption for this image.
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], caption)

        return conv

    def get_caption_prompt_rstr(self, caption=None) -> str:
        qs = "ryFo8ZVcyNMtLgryNOg64UTjySyEb79e5aq6IJxGuz0GzWNtoz. Provide a short caption for this image."
        # Consider thegivenimagebeingadversariallyperturbed. Provide a short caption for this image.
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], caption)

        return conv

    def get_caption_prompt_rsent(self, caption=None) -> str:
        qs = "Clouds drift quietly over the ancient, forgottencity.. Provide a short caption for this image."
        # Consider thegivenimagebeingadversariallyperturbed. Provide a short caption for this image.
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], caption)

        return conv

    def get_caption_prompt_corruptions_ap(self, corruption, caption=None) -> str:

        corruption_prompts = {
            "gaussian_noise": "Given the image may have Gaussian noise, it could appear grainy with random variations across the pixels.",
            "shot_noise": "Given the image may have shot noise, it could exhibit random bright or dark speckles, particularly noticeable in uniform areas.",
            "impulse_noise": "Given the image may have impulse noise, it could have sudden, random white or black pixel spikes.",
            "defocus_blur": "Given the image may have defocus blur, it could appear out of focus with a smooth loss of detail.",
            "glass_blur": "Given the image may have glass blur, it could appear distorted as though viewed through a textured or frosted glass.",
            "motion_blur": "Given the image may have motion blur, it could appear streaked or smeared, as if captured while in motion.",
            "zoom_blur": "Given the image may have zoom blur, it could appear stretched or radiate from the center, as if zoomed in rapidly.",
            "snow": "Given the image may have snow, it could be overlaid with white, flaky speckles, reducing visibility and simulating a snowy environment.",
            "frost": "Given the image may have frost, it could appear covered with a semi-transparent, icy pattern, reducing clarity.",
            "fog": "Given the image may have fog, it could appear hazy or have a washed-out look, reducing contrast and clarity.",
            "brightness": "Given the image's brightness may be altered, it could appear brighter or darker, affecting overall visibility.",
            "contrast": "Given the image's contrast may be altered, it could appear more vivid or more muted, affecting the distinction between light and dark areas.",
            "elastic_transform": "Given the image may have elastic transformation, it could appear warped or stretched in various directions, distorting the structure.",
            "pixelate": "Given the image may be pixelated, it could have reduced resolution, making details blocky and unclear.",
            "jpeg_compression": "Given the image may have JPEG compression artifacts, it could exhibit blocky distortions or loss of detail, especially around edges.",
            "speckle_noise": "Given the image may have speckle noise, it could appear with random granular spots, especially in smoother areas.",
            "gaussian_blur": "Given the image may have Gaussian blur, it could appear smoothly out of focus, losing sharp details uniformly.",
            "spatter": "Given the image may have spatter, it could have the appearance of splashed liquid or specks, distorting parts of the scene.",
            "saturate": "Given the image may have altered saturation, it could appear overly vivid or desaturated, affecting the intensity of colors."
        }

        qs = f"{corruption_prompts[corruption]} Provide a short caption for this image."

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], caption)

        return conv

    def get_caption_prompt_corruptions_ac(self, corruption, caption=None) -> str:

        corruption_prompts = {
            "gaussian_noise": "Given the image has been corrupted with Gaussian noise, it appears grainy with random variations across the pixels.",
            "shot_noise": "Given the image has been corrupted with shot noise, it exhibits random bright or dark speckles, particularly noticeable in uniform areas.",
            "impulse_noise": "Given the image has been corrupted with impulse noise, it has sudden, random white or black pixel spikes.",
            "defocus_blur": "Given the image has been corrupted with defocus blur, it appears out of focus with a smooth loss of detail.",
            "glass_blur": "Given the image has been corrupted with glass blur, it appears distorted as though viewed through a textured or frosted glass.",
            "motion_blur": "Given the image has been corrupted with motion blur, it appears streaked or smeared, as if captured while in motion.",
            "zoom_blur": "Given the image has been corrupted with zoom blur, it appears stretched or radiates from the center, as if zoomed in rapidly.",
            "snow": "Given the image has been corrupted with snow, it is overlaid with white, flaky speckles, reducing visibility and simulating a snowy environment.",
            "frost": "Given the image has been corrupted with frost, it appears covered with a semi-transparent, icy pattern, reducing clarity.",
            "fog": "Given the image has been corrupted with fog, it appears hazy or washed-out, reducing contrast and clarity.",
            "brightness": "Given the image has been corrupted with altered brightness, it appears either brighter or darker, affecting overall visibility.",
            "contrast": "Given the image has been corrupted with altered contrast, it appears either more vivid or more muted, affecting the distinction between light and dark areas.",
            "elastic_transform": "Given the image has been corrupted with elastic transformation, it appears warped or stretched in various directions, distorting the structure.",
            "pixelate": "Given the image has been corrupted with pixelation, it has reduced resolution, making details blocky and unclear.",
            "jpeg_compression": "Given the image has been corrupted with JPEG compression artifacts, it exhibits blocky distortions or loss of detail, especially around edges.",
            "speckle_noise": "Given the image has been corrupted with speckle noise, it appears with random granular spots, especially in smoother areas.",
            "gaussian_blur": "Given the image has been corrupted with Gaussian blur, it appears smoothly out of focus, losing sharp details uniformly.",
            "spatter": "Given the image has been corrupted with spatter, it looks as if splashed with liquid or specks, distorting parts of the scene.",
            "saturate": "Given the image has been corrupted with altered saturation, it appears overly vivid or desaturated, affecting the intensity of colors."
        }

        qs = f"{corruption_prompts[corruption]} Provide a short caption for this image."

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], caption)

        return conv

    def get_caption_prompt_mt(self, caption=None) -> str:
        qs = "Please provide a detailed description of the image."

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], caption)

        return conv
