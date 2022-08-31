#!/bin/bash

mkdir -p bert-base
cd bert-base
wget https://huggingface.co/bert-base-uncased/resolve/main/config.json
wget https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
wget https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json
wget https://huggingface.co/bert-base-uncased/resolve/main/tokenizer_config.json
wget https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
cd -

mkdir -p roberta-base
cd roberta-base
wget https://huggingface.co/roberta-base/resolve/main/config.json
wget https://huggingface.co/roberta-base/resolve/main/dict.txt
wget https://huggingface.co/roberta-base/resolve/main/merges.txt
wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
wget https://huggingface.co/roberta-base/resolve/main/tokenizer.json
wget https://huggingface.co/roberta-base/resolve/main/vocab.json

cd -