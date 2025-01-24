#!/bin/bash

baseModel='LLAVA'
DATA_ROOT=${1:-/path/to/eval_benchmark}
MODEL_PATH=${2:-liuhaotian/llava-v1.5-7b}
load_encoder=${3:-none}
EXP_NUM=${4:-1}

# baseModel='openFlamingo'

#modelPath=${1}
#if [ -z "${modelPath}" ]
#then
#      echo "\$modelPath is empty Using robust model from here: "
#      modelPath=/path/to/ckpt.pt
#      modelPath1=ckpt_name
#else
#      echo "\$modelPath is NOT empty"
#      modelPath1=${modelPath}
#fi
#
#answerFile="${baseModel}_${modelPath1}"
#echo "Will save to the following json: "
#echo $answerFile

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --load_encoder $load_encoder \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder $DATA_ROOT/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-check_${EXP_NUM}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-check_${EXP_NUM}.jsonl \
    --model_path $MODEL_PATH \
    --load_encoder $load_encoder