#!/bin/bash

DATA_ROOT=${1:-/mnt/nvme0n1/Dataset/muzammal/VLM_datasets/LLaVA/eval_benchmark}
MODEL_PATH=${2:-liuhaotian/llava-v1.5-7b}
load_encoder=${3:-none}
DEVICE=${4:-2}
flickr=${5:-false}
coco=${6:-false}
vqav2=${7:-false}
textvqa=${8:-false}
vizwiz=${9:-false}
okvqa=${10:-false}
mt_trust=${11:-false}
attack=${12:-none}
eps=${13:-2}

# echo all arguments
echo "All arguments: $@"
echo "To not save adversarial examples add --dont_save_adv flag to the command"

# LLaVA evaluation script
CUDA_VISIBLE_DEVICES=$DEVICE python -m vlm_eval.run_evaluation \
--load_encoder $load_encoder \
--eval_flickr $flickr \
--eval_coco $coco \
--eval_vqav2 $vqav2 \
--eval_textvqa $textvqa \
--eval_vizwiz $vizwiz \
--eval_ok_vqa $okvqa \
--eval_multitrust $mt_trust \
--attack $attack --eps $eps --steps 100 --mask_out none \
--vision_encoder_pretrained none \
--precision float16 \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file llava \
--model llava \
--temperature 0.0 \
--num_beams 1 \
--out_save_path $MODEL_PATH \
--out_base_path ./results_debug \
--model_path $MODEL_PATH \
--coco_train_image_dir_path $DATA_ROOT/coco/train2014 \
--coco_val_image_dir_path $DATA_ROOT/coco/val2014 \
--coco_karpathy_json_path $DATA_ROOT/coco/annotations/karpathy_coco.json \
--coco_annotations_json_path $DATA_ROOT/coco/annotations/captions_val2014.json \
--flickr_image_dir_path $DATA_ROOT/flickr30k/flickr30k-images \
--flickr_karpathy_json_path $DATA_ROOT/flickr30k/karpathy_flickr30k.json \
--flickr_annotations_json_path $DATA_ROOT/flickr30k/dataset_flickr30k_coco_style.json \
--vizwiz_train_image_dir_path $DATA_ROOT/vizwiz/train \
--vizwiz_test_image_dir_path $DATA_ROOT/vizwiz/val \
--vizwiz_train_questions_json_path $DATA_ROOT/vizwiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path $DATA_ROOT/vizwiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path $DATA_ROOT/vizwiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path $DATA_ROOT/vizwiz/val_annotations_vqa_format.json \
--vqav2_train_image_dir_path $DATA_ROOT/coco/train2014 \
--vqav2_train_questions_json_path $DATA_ROOT/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \
--vqav2_train_annotations_json_path $DATA_ROOT/vqav2/v2_mscoco_train2014_annotations.json \
--vqav2_test_image_dir_path $DATA_ROOT/coco/val2014 \
--vqav2_test_questions_json_path $DATA_ROOT/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
--vqav2_test_annotations_json_path $DATA_ROOT/vqav2/v2_mscoco_val2014_annotations.json \
--ok_vqa_train_image_dir_path $DATA_ROOT/coco/train2014 \
--ok_vqa_train_questions_json_path $DATA_ROOT/okvqa/OpenEnded_mscoco_train2014_questions.json \
--ok_vqa_train_annotations_json_path $DATA_ROOT/okvqa/mscoco_train2014_annotations.json \
--ok_vqa_test_image_dir_path $DATA_ROOT/coco/val2014 \
--ok_vqa_test_questions_json_path $DATA_ROOT/okvqa/OpenEnded_mscoco_val2014_questions.json \
--ok_vqa_test_annotations_json_path $DATA_ROOT/okvqa/mscoco_val2014_annotations.json \
--textvqa_image_dir_path $DATA_ROOT/textvqa/train_images \
--textvqa_train_questions_json_path $DATA_ROOT/textvqa/train_questions_vqa_format.json \
--textvqa_train_annotations_json_path $DATA_ROOT/textvqa/train_annotations_vqa_format.json \
--textvqa_test_questions_json_path $DATA_ROOT/textvqa/val_questions_vqa_format.json \
--textvqa_test_annotations_json_path $DATA_ROOT/textvqa/val_annotations_vqa_format.json \
--mt_data_path $DATA_ROOT/MultiTrust/robustness/adv_nips
