# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This code is modified based on the official implementation in the SentEval toolkit
credit to: https://github.com/facebookresearch/SentEval/blob/main/examples/bow.py
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import argparse


# Set PATHs
PATH_TO_SENTEVAL = '../../SentEval'
PATH_TO_DATA = '../../SentEval/data'
PATH_TO_VEC = '../../dataset/GLOVE/glove.840B.300d.txt'


# Set logging information
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    
    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Logging function
def custom_log(logs):
        """
        Log metrics helper function
        """
        def log_format(logs):
            """
            Reformat Trainer logs to human readable format
            """
            format_logs = logs.copy()
            
            for k, v in format_logs.items():
                if 'loss' in k:
                    format_logs[k] = round(v, 4)
                elif 'epoch' in k:
                    format_logs[k] = round(v, 2)
                elif 'eval' in k:
                    format_logs[k] = round(v, 3)
                elif '_rate' in k:
                    format_logs[k] = float(format(v, '.4g'))

            return format_logs

        _ = logs.pop("total_flos", None)
        
        logger.info(f"******** Logging Info ********")
        logs_formatted = log_format(logs)
        k_width = max(len(str(x)) for x in logs_formatted.keys())
        v_width = max(len(str(x)) for x in logs_formatted.values())
        for key in sorted(logs_formatted.keys()):
            logger.info(f"  {key: <{k_width}} = {logs_formatted[key]:>{v_width}}")


params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target task to be evaluated by GloVe embedding")
    parser.add_argument('--target_task', help="Target task to be evaluated", type=str, required=True)
    args = parser.parse_args()
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']
    # tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
    #                 'Length', 'WordContent', 'Depth',
    #                 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 
    #                 'CoordinationInversion']
    # tasks = ['WILA', 'WILB', 'WILC']
    # tasks = ['HyperParNews', 'IMDB']
    tasks = [args.target_task]
    results = se.eval(tasks)
    metrics = {}

    # Print all logs
    for task in tasks:
        if task in ['WILA', 'WILB', 'WILC', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC', 'Tense', 'SICKEntailment', 'Length', 'WordContent', 'Depth',
                    'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 
                    'CoordinationInversion']:
            metrics['eval_{}_test_std'.format(task)] = results[task]['acc']
            metrics['eval_{}_dev_std'.format(task)] = results[task]['devacc']
        elif task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                metrics['eval_{}_test_std'.format(task)] = results[task]['all']['spearman']['all'] * 100
                metrics['eval_{}_dev_std'.format(task)] = results[task]['dev']['spearman'][0] * 100
            else:
                metrics['eval_{}_dev_std'.format(task)] = results[task]['dev']['spearman'][0] * 100
                metrics['eval_{}_test_std'.format(task)] = results[task]['test']['spearman'].correlation * 100
    
    custom_log(metrics)
