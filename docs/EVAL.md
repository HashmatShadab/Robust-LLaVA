We conduct a comprehensive evaluation of models across a diverse set of benchmarks. In addition to assessing model performance on
clean examples, as done in the original [LLaVA](https://github.com/haotian-liu/LLaVA) codebase, we extend the evaluation to include both untargeted and targeted attacks on benchmark datasets. Furthermore, we analyze model robustness against common corruptions to provide a more holistic assessment of their reliability.

## Data Preparation

Before preparing task-specific data, **you MUST first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**. It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to `./playground/data/eval`. This also provides a general structure for all datasets.

### Data Structure
The data for evaluation can be stored in the `eval_benchmark` directory. The structure of the  directory will be as follows:
```
$DATA_ROOT
├── coco
│   ├── train2014
│   ├── val2014
│   ├── annotations
│   │   ├── karpathy_coco.json
│   │   ├── captions_val2014.json
│   │   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   │   ├── v2_mscoco_train2014_annotations.json
│   │   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   │   ├── v2_mscoco_val2014_annotations.json
│   │   ├── OpenEnded_mscoco_train2014_questions.json
│   │   ├── mscoco_train2014_annotations.json
│   │   ├── OpenEnded_mscoco_val2014_questions.json
│   │   ├── mscoco_val2014_annotations.json
│   │
├── flickr30k
│   ├── flickr30k-images
│   ├── karpathy_flickr30k.json
│   ├── dataset_flickr30k_coco_style.json
│
├── vizwiz
│   ├── train
│   ├── val
│   ├── train_questions_vqa_format.json
│   ├── train_annotations_vqa_format.json
│   ├── val_questions_vqa_format.json
│   ├── val_annotations_vqa_format.json
│
├── vqav2
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   ├── v2_mscoco_train2014_annotations.json
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   ├── v2_mscoco_val2014_annotations.json
│
├── okvqa
│   ├── OpenEnded_mscoco_train2014_questions.json
│   ├── mscoco_train2014_annotations.json
│   ├── OpenEnded_mscoco_val2014_questions.json
│   ├── mscoco_val2014_annotations.json
│
├── textvqa
│   ├── train_images
│   ├── train_questions_vqa_format.json
│   ├── train_annotations_vqa_format.json
│   ├── val_questions_vqa_format.json
│   ├── val_annotations_vqa_format.json


```
The annotation files for each can be found in the huggingface datasets [eval_benchmark](https://huggingface.co/datasets/openflamingo/eval_benchmark/tree/main/mscoco_karpathy) and can be kept in the above structure or you can 
download the datasets/annotations from this [Link](https://drive.google.com/drive/folders/1Ipb6o3g-9ll2g7rP8dfwS2oOiIOcu4nU?usp=sharing) and keep it in the above structure. Clean evaluation can be done on more benchmarks as provided
in the original [LLaVA](https://github.com/haotian-liu/LLaVA) codebase, but for adversarial evaluation, we have provided the evaluation on above benchmarks.

## Evaluation

---
First download the instruction tuned model checkpoints from Model Zoo as mentioned in the [README](../README.md). In the below scripts, following variables should be set:

- **`$DATA_ROOT`**: Path to the root directory of the evaluation data.
- **`$MODEL_PATH`**: Path to the model checkpoint.
- **`$ENCODER`**:
  - Set to `'simclip4'` when using the **adversarially finetuned CLIP model from SimCLIP-4**, replacing the original CLIP model in the LLaVA framework.
  - Set to `'fare4'` when using the **adversarially finetuned CLIP model from FARE-4**, replacing the original CLIP model in the LLaVA framework.
  - Set to `'none'` for all other cases.
- **`$GPU`**: ID of the GPU to be used.
- **`$EPSILON`**: Perturbation budget for adversarial evaluation.
  - For example, to evaluate on a perturbation budget of **2/255**, set `$EPSILON=2`.
---
    
### Flickr30k
To evaluate the adversarial performance(APGD-Ensemble Attack) of model on Flickr, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU true false false false false false false ensemble $EPSILON
```


To evaluate the adversarial performance(APGD Attack) of model on Flickr, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU true false false false false false false apgd $EPSILON
```

To evaluate the clean performance of model on Flickr, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU true false false false false false false none 
```
---


### COCO
To evaluate the adversarial performance(APGD-Ensemble Attack) of model on COCO, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false true false false false false false ensemble $EPSILON
```

To evaluate the adversarial performance(APGD Attack) of model on COCO, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false true false false false false false apgd $EPSILON
```

To evaluate the clean performance of model on COCO, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false true false false false false false none 
```
---

### VQAv2
To evaluate the adversarial performance(APGD-Ensemble Attack) of model on VQAv2, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false true false false false false ensemble $EPSILON
```

To evaluate the adversarial performance(APGD Attack) of model on VQAv2, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false true false false false false apgd $EPSILON
```

To evaluate the clean performance of model on VQAv2, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false true false false false false none 
```
---

### TextVQA
To evaluate the adversarial performance(APGD-Ensemble Attack) of model on TextVQA, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false true false false false ensemble $EPSILON
```

To evaluate the adversarial performance(APGD Attack) of model on TextVQA, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false true false false false apgd $EPSILON
```

To evaluate the clean performance of model on TextVQA, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false true false false false none 
```
---

### VizWiz
To evaluate the adversarial performance(APGD-Ensemble Attack) of model on VizWiz, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false true false false ensemble $EPSILON
```

To evaluate the adversarial performance(APGD Attack) of model on VizWiz, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false true false false apgd $EPSILON
```

To evaluate the clean performance of model on VizWiz, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false true false false none 
```
---

### OKVQA
To evaluate the adversarial performance(APGD-Ensemble Attack) of model on OKVQA, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false true false ensemble $EPSILON
```

To evaluate the adversarial performance(APGD Attack) of model on OKVQA, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false true false apgd $EPSILON
```

To evaluate the clean performance of model on OKVQA, run the following command:
```bash
bash bash/llava_eval.sh $DATA_ROOT $MODEL_PATH $ENCODER $GPU false false false false false true false none 
```
---

### POPE
To evaluate the performance of model on POPE, run the following command:
```bash
bash bash/eval_pope.sh $DATA_ROOT $MODEL_PATH $ENCODER
```  
---

To run clean evaluation on all the benchmarks, run the following command:
```bash
bash bash/clean_evaluation.sh $MODEL_PATH $DATA_ROOT $GPU $ENCODER
```

To run adversarial evaluation on all the benchmarks, run the following command:
```bash
bash bash/adv_evaluation.sh $MODEL_PATH $DATA_ROOT $GPU $ENCODER $EPSILON
```

where `$MODEL_PATH` is the path to the model checkpoint, `$DATA_ROOT` is the path to the data directory, `$GPU` is the GPU ID, and `$ENCODER` is the encoder type. For example, to evaluate the model on all benchmarks with an epsilon of 2/255, set `$EPSILON=2`.

---