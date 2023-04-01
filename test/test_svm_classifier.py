from classification.svm_classifier import SVMClassifier
from classification.dataset import Dataset, DatasetSplit

import pytest
import pathlib
import tempfile
import os
import pdb

class TestSVMClassifier:

    def test_train(self, imdb):
        classifier = SVMClassifier.train(imdb['train'])
        assert classifier.evaluate(imdb['train'])['f'] >= 0.95
        assert classifier.evaluate(imdb['dev'])['f'] >= 0.85
        assert classifier.evaluate(imdb['test'])['f'] >= 0.85
        
    def test_search_hyperparameters(self, imdb):
        params = SVMClassifier.search_hyperparameters(imdb['train'], imdb['dev'], n_trials=10)
        assert params['min_df'] >= 1
        assert params['max_df'] < 1.0
        assert params['loss'] in ['hinge', 'squared_hinge']
        assert 0.25 <= params['c'] <= 2.0
        assert 500 <= params['max_iter'] <= 1500
        assert params['top_k'] is None
        params = SVMClassifier.search_hyperparameters(imdb['train'], imdb['dev'], n_trials=1, search_top_k=True)
        assert 1 <= params['top_k'] < 2

    def test_train_with_kwargs(self, imdb):
        params = {
            'min_df': 3,
            'max_df': 0.8,
            'loss': 'hinge',
            'c': 1.5,
            'max_iter': 100,
            'top_k': None,
        }
        model = SVMClassifier.train(imdb['train'], **params)
        with pytest.raises(ValueError):
            params['max_df'] = 2
            SVMClassifier.train(imdb['train'], **params)

    def test_topk(self, clothes):
        params = SVMClassifier.search_hyperparameters(clothes['train'], clothes['dev'], n_trials=1, search_top_k=True)
        assert 1 <= params['top_k'] < 5
        classifier = SVMClassifier.train(clothes['train'], top_k=3)
        
        
