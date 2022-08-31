#!/bin/bash

PLM_NAME=bert
TASK=HyperParNews            # When TASK is SICK related tasks, set the datadir to SICK
BATCH_SIZE=1
LR=1e-0                           # 3e-5
PROMPT_LENGTH=1               # 2
NUM_TRAIN_EPOCHS=100
MLM_PROB=0.5                     # 0.25
LR_SCHEDULER=linear
OPTIMIZER_NAME=sgd
WEIGHT_DECAY=0.00
WARMUP_RATIO=0.1
MASK_SCHEME=random\_mask
DUPLICATE_TIMES=64              # 16
SEED=16

echo "Evaluate MLM loss/acc $PLM_NAME-$TASK $NUM_TRAIN_EPOCHS epochs with hyperparameters lr $LR, lr_scheduler $LR_SCHEDULER bs $BATCH_SIZE, prompt_length $PROMPT_LENGTH, mlm_probability $MLM_PROB, mask_scheme $MASK_SCHEME"

python train_document_mlm.py \
    --model_name_or_path ../pretrained_models/$PLM_NAME-base \
    --train_file ../dataset/$TASK/raw_data.txt \
    --cache_dir ../cache_dir \
    --output_dir ../checkpoints/$PLM_NAME-base\_$TASK/doc-eval-optimizer-$OPTIMIZER_NAME-seed-$SEED-my-mlm_prob-$MLM_PROB-lr-$LR-duptime-$DUPLICATE_TIMES-mask_scheme-$MASK_SCHEME-lossacc-trainepoch-$NUM_TRAIN_EPOCHS-prompt_length-$PROMPT_LENGTH-scheduler-$LR_SCHEDULER-warmmup-$WARMUP_RATIO-weightdecay-$WEIGHT_DECAY-batch_size-$BATCH_SIZE \
    --overwrite_output_dir \
    --line_by_line \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 64 \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --max_seq_length 256 \
    --plm_name $PLM_NAME \
    --lm_head_name $MASK_SCHEME \
    --mlm_probability $MLM_PROB \
    --sent_embed_size 768 \
    --prompt_length $PROMPT_LENGTH \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_strategy epoch \
    --save_total_limit 2 \
    --lr_scheduler_type $LR_SCHEDULER \
    --optimizer_name $OPTIMIZER_NAME \
    --metric_for_best_model $TASK\_devacc\_onlysent_fast \
    --validation_split_percentage 0 \
    --mlm_loss_no_reduction \
    --init_with_zero \
    --eval_every_k_epochs 1 \
    --eval_after_k_epochs 0 \
    --early_stopping_patience 20 \
    --seed $SEED \
    --fp16 \
    --target_task $TASK \
    --duplicate_times $DUPLICATE_TIMES \
    --select_span_length 256 
    # --resume_from_checkpoint ../checkpoints/$PLM_NAME-base\_$TASK/doc-eval-optimizer-$OPTIMIZER_NAME-seed-$SEED-my-mlm_prob-$MLM_PROB-lr-$LR-duptime-$DUPLICATE_TIMES-mask_scheme-$MASK_SCHEME-lossacc-trainepoch-$NUM_TRAIN_EPOCHS-prompt_length-$PROMPT_LENGTH-scheduler-$LR_SCHEDULER-warmmup-$WARMUP_RATIO-weightdecay-$WEIGHT_DECAY-batch_size-$BATCH_SIZE/checkpoint-120000
    # all masks as input: eval-mlmmy-lossacc-all_mask_input-trainepoch-$NUM_TRAIN_EPOCHS-prompt_length-$PROMPT_LENGTH-lr-$LR-scheduler-$LR_SCHEDULER-weightdecay-$WEIGHT_DECAY-batch_size-$BATCH_SIZE-mlm_prob-$MLM_PROB
    # --resume_from_checkpoint ../checkpoints/$PLM_NAME-base\_$TASK/eval-mlmmy-lossacc-trainepoch-500-prompt_length-2-lr-5e-4-scheduler--weightdecay-0.00-batch_size-32-mlm_prob-0.25/checkpoint-382500 \
    # metric_for_best_model $TASK\_$EVAL_TARGET\_devacc \ / accuracy
    # --use_reduce_on_plateau_scheduler
    # --init_with_zero \
    # ../pretrained_models/$PLM_NAME-base \
    # --model_name_or_path ../pretrained_models/$PLM_NAME-base \
    # --model_name_or_path ../checkpoints/$PLM_NAME-base\_$TASK/doc-eval-optimizer-$OPTIMIZER_NAME-seed-$SEED-my-mlm_prob-$MLM_PROB-lr-$LR-duptime-$DUPLICATE_TIMES-mask_scheme-$MASK_SCHEME-lossacc-trainepoch-$NUM_TRAIN_EPOCHS-prompt_length-$PROMPT_LENGTH-scheduler-$LR_SCHEDULER-warmmup-$WARMUP_RATIO-weightdecay-$WEIGHT_DECAY-batch_size-$BATCH_SIZE/checkpoint-29670 \