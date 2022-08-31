# SentencePrompt
Source code for the Imperial College London Msc Individual Project "Capturing the Gap in Pre-trained Language Models"

## Install dependencies
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Download the Pretrain Language Models
Starting from the root dir `SentencePrompt/', run
```bash
cd pretrained_models
sh download_pretrained_models.sh
```

## Download SentEval data and our processed data (one sentence a line)
Starting from the root dir `SentencePrompt/', run
```bash
cd SentEval/data
sh download_senteval_data.sh
tar xvfz senteval_data.tar.gz
rm -f senteval_data.tar.gz
```
Then return to the root dir and enter `SentencePrompt/dataset/`, run
```bash
cd dataset
sh download_all_dataset.sh
tar xvfz processed_data.tar.gz
rm -f processed_data.tar.gz
```

## Train and Evaluate on Downstream, Probing tasks
First enter `src` folder and run
```bash
sh run_eval_batch_mlm.sh
```
Here `PLM_NAME` is 'bert' or 'roberta',`TASK` can be selected among `['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']`.

## Train and Evaluate on Long Document Classification tasks
First enter `src` folder and run
```bash
sh run_eval_doc_mlm.sh
```
Here `PLM_NAME` is 'bert' or 'roberta',`TASK` can be selected among `[HyperParNews, IMDB]`.

## Run the baselines
Go to `src/baselines/` folder and run:
+ GloVe baselines
```bash
python eval_glove_emb_avg.py --target_task CR
```
Here CR is the target task.

+ BERT/RoBERTa baselines for downstream and probing tasks
```bash
sh run_eval_batch_bert_plain.sh
```
Here `PLM_NAME` is 'bert' or 'roberta',`TASK` can be selected among `['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']`, `model_name_or_path` is the dir for the pre-trained language model, `evaluate_mode` is the pooling method to be used, which can be selected in `[cls, avg, concat_first_last, concat_with_zero, concat_with_random]`.

+ BERT/RoBERTa/Longformer baselines for long document classification tasks
```bash
sh run_eval_doc_bert_plain.sh
```
Here `PLM_NAME` is 'bert' or 'roberta',`TASK` can be selected among `[HyperParNews, IMDB]`, `model_name_or_path` is the dir for the pre-trained language model, `evaluate_mode` is the pooling method to be used, which can be selected in `[avg, concat_first_last, concat_with_zero, concat_with_random]`. To evaluate on longformer, uncomment the line `--use_longformer`.

## Acknowledgement
Our `train_*` scripts are modified based on the example scripts which can be found at https://github.com/huggingface/transformers/blob/v4.18-release/examples/pytorch/language-modeling/run_mlm.py.

We add our own SentEval evaluation scripts based on the original SentEval repo which can be found at https://github.com/facebookresearch/SentEval.
