'''
Document binary classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from senteval.tools.validation import KFoldClassifier
from senteval.tools.validation import InnerKFoldClassifier


class DocumentEval(object):
    def __init__(self, task_path, seed=1111):
        self.seed = seed
        self.task_name = task_path.split('downstream/')[-1]
        logging.info(f'***** Document classification task : {self.task_name} *****\n\n')
        self.train = self.loadFile(os.path.join(task_path, 'train.txt'))
        self.test = self.loadFile(os.path.join(task_path, 'test.txt'))

    def do_prepare(self, params, prepare):
        # Prepare data
        samples = self.train['X'] + self.test['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        ged_data = {'X': [], 'y': []}
        
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                target, sample = line.strip().split('|||', 1)
                target = int(target)
                sample = sample.strip().split()
                ged_data['X'].append(sample)
                ged_data['y'].append(target)
        return ged_data

    def run(self, params, batcher):
        train_embeddings, test_embeddings = [], []

        # Sort to reduce padding
        sorted_corpus_train = sorted(zip(self.train['X'], self.train['y']),
                                     key=lambda z: (len(z[0]), z[1]))
        train_samples = [x for (x, y) in sorted_corpus_train]
        train_labels = [y for (x, y) in sorted_corpus_train]

        sorted_corpus_test = sorted(zip(self.test['X'], self.test['y']),
                                    key=lambda z: (len(z[0]), z[1]))
        test_samples = [x for (x, y) in sorted_corpus_test]
        test_labels = [y for (x, y) in sorted_corpus_test]

        # Get train embeddings
        for ii in range(0, len(train_labels), params.batch_size):
            batch = train_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            train_embeddings.append(embeddings)
        train_embeddings = np.vstack(train_embeddings)
        logging.info('Computed train embeddings')

        # Get test embeddings
        for ii in range(0, len(test_labels), params.batch_size):
            batch = test_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            test_embeddings.append(embeddings)
        test_embeddings = np.vstack(test_embeddings)
        logging.info('Computed test embeddings')

        config_classifier = {'nclasses': 2, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'kfold': params.kfold}
        clf = KFoldClassifier({'X': train_embeddings,
                               'y': np.array(train_labels)},
                              {'X': test_embeddings,
                               'y': np.array(test_labels)},
                              config_classifier)
        devacc, testacc, _ = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} \
            for Doc classification\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.train['X']), 'ntest': len(self.test['X'])}


class DocumentInnerKfoldEval(object):
    def __init__(self, task_path, seed=1111):
        self.seed = seed
        self.task_name = task_path.split('downstream/')[-1]
        logging.info(f'***** Document classification task : {self.task_name} *****\n\n')
        self.data = self.loadFile(os.path.join(task_path, 'all.txt'))

    def do_prepare(self, params, prepare):
        # prepare is given the whole text
        return prepare(params, self.data['X'])
        # prepare puts everything it outputs in "params" : params.word2id etc
        # Those output will be further used by "batcher".

    def loadFile(self, fpath):
        doc_data = {'X': [], 'y': []}
        
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                target, sample = line.strip().split('|||', 1)
                target = int(target)
                sample = sample.strip().split()
                doc_data['X'].append(sample)
                doc_data['y'].append(target)
        return doc_data

    def run(self, params, batcher):
        all_embeddings = []

        # Sort to reduce padding
        sorted_corpus = sorted(zip(self.data['X'], self.data['y']),
                                     key=lambda z: (len(z[0]), z[1]))
        all_samples = [x for (x, y) in sorted_corpus]
        all_labels = [y for (x, y) in sorted_corpus]

        # Get train embeddings
        logging.info('Generating document embeddings')
        for ii in range(0, len(all_labels), params.batch_size):
            batch = all_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            all_embeddings.append(embeddings)
        all_embeddings = np.vstack(all_embeddings)
        logging.info('Generated document embeddings')

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = InnerKFoldClassifier(all_embeddings, np.array(all_labels), config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1}\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'ndev': len(self.data['X']),
                'ntest': len(self.data['X'])}