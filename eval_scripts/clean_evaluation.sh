#!/bin/bash

MODEL_PATH=${1:-liuhaotian/llava-v1.5-7b}
DATA_ROOT=${2:-/path/to/eval/benchmark}
GPU=${3:-3}
ENCODER=${4:-none}
EPSILON=${5:-2}


# bash script.sh model_path load_encoder GPU flickr coco vqav2 textvqa vizwiz okvqa multitrust attack eps

bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU true false false false false false false none $EPSILON

bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false true false false false false false none $EPSILON

bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false true false false false false none $EPSILON

bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false true false false false none $EPSILON

bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false true false false none $EPSILON

bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false true false none $EPSILON

bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false false true none $EPSILON


bash bash/eval_pope.sh $DATA_ROOT  $MODEL_PATH $ENCODER



