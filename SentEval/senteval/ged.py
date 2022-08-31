'''
Grammar Error Detection binary classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from senteval.tools.validation import KFoldClassifier
from collections import defaultdict


class GEDEvalErrorType(object):
    def __init__(self, task_path, seed=1111):
        self.seed = seed
        self.task_name = task_path.split('downstream/')[-1]
        logging.info(f'***** Grammar task : {self.task_name} *****\n\n')
        self.train = self.loadFile(os.path.join(task_path, 'train_finegrained.txt'))
        self.test = self.loadFile(os.path.join(task_path, 'test_finegrained.txt'))
        self.error_genres_dict = {'M':[], 'U':[], 'R':[], 'UNK':[]}
        self.error_types_dict = {
            'ADJ':[], 'ADJ:FORM':[], 'ADV':[], 'CONJ':[], 'CONTR':[], 'DET':[], 'MORPH':[], 'NOUN':[],
            'NOUN:INFL':[], 'NOUN:NUM':[], 'NOUN:POSS':[], 'ORTH':[], 'OTHER':[], 'PART':[], 'PREP':[],
            'PRON':[], 'PUNCT':[], 'SPELL':[], 'VERB':[], 'VERB:FORM':[], 'VERB:INFL':[], 'VERB:SVA':[], 'VERB:TENSE':[], 'WO':[]
        }
        self.error_genres_types_dict = defaultdict(list)

    def do_prepare(self, params, prepare):
        # Prepare data
        samples = self.train['X'] + self.test['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        ged_data = {'X': [], 'y': [], 'error_genre': [], 'error_type': []}
        
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                target, sample, error_genre, error_type = line.strip().split('|||')
                target = int(target)
                sample = sample.strip().split()
                error_genre = error_genre.strip().split('-')
                error_type = error_type.strip().split('-')
                ged_data['X'].append(sample)
                ged_data['y'].append(target)
                ged_data['error_genre'].append(error_genre)
                ged_data['error_type'].append(error_type)
        return ged_data

    def run(self, params, batcher):
        train_embeddings, test_embeddings = [], []

        # Sort to reduce padding
        sorted_corpus_train = sorted(zip(self.train['X'], self.train['y'], self.train['error_genre'], self.train['error_type']),
                                     key=lambda z: (len(z[0]), z[1]))
        train_samples = [x for (x, y, g, t) in sorted_corpus_train]
        train_labels = [y for (x, y, g, t) in sorted_corpus_train]
        train_genres = [g for (x, y, g, t) in sorted_corpus_train]
        train_types = [t for (x, y, g, t) in sorted_corpus_train]

        sorted_corpus_test = sorted(zip(self.test['X'], self.test['y'], self.test['error_genre'], self.test['error_type']),
                                    key=lambda z: (len(z[0]), z[1]))
        test_samples = [x for (x, y, g, t) in sorted_corpus_test]
        test_labels = [y for (x, y, g, t) in sorted_corpus_test]
        test_genres = [g for (x, y, g, t) in sorted_corpus_test]
        test_types = [t for (x, y, g, t) in sorted_corpus_test]

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
        # Here I need to return the prediction for each sample
        devacc, testacc, test_predicts = clf.run()
        res_genre_score, res_type_score, res_genre_type_score = self.error_genre_type_score(test_predicts, test_labels, test_genres, test_types)
        # self.save_predict_result(test_samples, test_predicts, test_labels, test_genres, test_types)
        logging.debug('\nDev acc : {0} Test acc : {1} \
            for GED\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.train['X']), 'ntest': len(self.test['X']),
                'error_genre_acc': res_genre_score, 'error_type_acc': res_type_score, 'error_genre_type_acc': res_genre_type_score}

    def error_genre_type_score(self, test_predicts, test_labels, test_error_genres, test_error_types):
        # Count the total number of each genre / type and the number of correct predictions under that genre / type
        test_predicts = test_predicts.squeeze().tolist()
        for i, error_genres in enumerate(test_error_genres):
            test_predict, test_label = int(test_predicts[i]), test_labels[i]
            error_types = test_error_types[i]
            correctness = int(test_predict == test_label)
            try:
                for error_genre in error_genres:
                    self.error_genres_dict[error_genre].append(correctness)
                for error_type in error_types:
                    if error_type != '':
                        self.error_types_dict[error_type].append(correctness)
                # genre:type
                for error_genre in error_genres:
                    for error_type in error_types:
                        if error_type == '':
                            error_type = 'NOUN'
                        self.error_genres_types_dict[f'{error_genre}:{error_type}'].append(correctness)
            except Exception as e:
                raise ValueError('Wrong error type or genre input!')

        # Compute the acc for each error genre / type
        # Each item in the dict is a tuple (accuracy, num of one wrong type/genre)
        res_genre_dict, res_type_dict, res_genre_type_dict = dict(), dict(), dict()
        for error_genre in self.error_genres_dict:
            tmp_list = self.error_genres_dict[error_genre]
            res_genre_dict[error_genre] = (round(100.0 * sum(tmp_list) / len(tmp_list), 2), len(tmp_list))
        for error_type in self.error_types_dict:
            tmp_list = self.error_types_dict[error_type]
            try:
                res_type_dict[error_type] = (round(100.0 * sum(tmp_list) / len(tmp_list), 2), len(tmp_list))
            except Exception as e:
                res_type_dict[error_type] = (0.00, len(tmp_list))
        for error_gt in self.error_genres_types_dict:
            tmp_list = self.error_genres_types_dict[error_gt]
            res_genre_type_dict[error_gt] = (round(100.0 * sum(tmp_list) / len(tmp_list), 2), len(tmp_list))
        # import pdb
        # pdb.set_trace()
        # aa = 'file.pkl'
        # import pickle
        # with open(aa, 'wb') as f:
        #     pickle.dump(self.error_types_dict, f)
        return res_genre_dict, res_type_dict, res_genre_type_dict


    def save_predict_result(self, test_samples, test_predicts, test_labels, test_genres, test_types):
        file_name = 'aa'
        import csv
        import pdb
        pdb.set_trace()
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            head = ['sentence', 'predict', 'label', 'correctness', 'error_genre', 'error_type']
            writer.writerow(head)
            for sample, predict, label, genre, error_type in zip(test_samples, test_predicts, test_labels, test_genres, test_types):
                sample = ' '.join(sample)
                predict, label = int(predict), int(label)
                writer.writerow([sample, predict, label, int(predict == label), genre, error_type])


class GEDEval(object):
    def __init__(self, task_path, seed=1111):
        self.seed = seed
        self.task_name = task_path.split('downstream/')[-1]
        logging.info(f'***** Grammar task : {self.task_name} *****\n\n')
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
            for GED\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.train['X']), 'ntest': len(self.test['X'])}
