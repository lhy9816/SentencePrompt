#!/bin/bash

PLM_NAME=roberta
TASK=HyperParNews

python eval_bert_doc_plain.py \
    --model_name_or_path ../../pretrained_models/$PLM_NAME-base \
    --plm_name $PLM_NAME \
    --evaluate_mode concat_first_last \
    --target_task $TASK \
    --cache_dir ../../cache_dir \
    --max_seq_length 256
    # --use_longformer