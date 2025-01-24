#!/bin/bash

MODEL_PATH=${1:-liuhaotian/llava-v1.5-7b}
DATA_ROOT=${2:-/path/to/eval_benchmark}
GPU=${3:-3}
ENCODER=${4:-none}
EPSILON=${5:-2}



bash bash/llava_eval_corruptions.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false true false false false false false none $EPSILON

bash bash/llava_eval_corruptions.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false true false false false false none $EPSILON


bash bash/llava_eval_corruptions.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false true false false none $EPSILON

bash bash/llava_eval_corruptions.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false true false none $EPSILON


