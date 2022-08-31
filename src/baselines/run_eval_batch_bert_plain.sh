#!/bin/bash

PLM_NAME=bert
TASK=WILA

python eval_bert_batch_plain.py \
    --model_name_or_path ../../pretrained_models/$PLM_NAME-base \
    --plm_name $PLM_NAME \
    --evaluate_mode concat_first_last \
    --target_task $TASK \
    --cache_dir ../../cache_dir