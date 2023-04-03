from classification.svm_classifier import SVMClassifier

import pytest
import pdb

class TestSVMClassifier:

    def test_train_binary(self, imdb):
        
        classifier = SVMClassifier.train(imdb['train'])
        assert 0.99 <= classifier.evaluate(imdb['train'])['f'] <= 1.00
        assert 0.90 <= classifier.evaluate(imdb['dev'])['f'] <= 0.91
        assert 0.85 <= classifier.evaluate(imdb['test'])['f'] <= 0.86

    def test_train_multiclass(self, clothes):
        classifier = SVMClassifier.train(clothes['train'])
        assert 0.94 <= classifier.evaluate(clothes['train'])['f'] <= 0.95
        assert 0.62 <= classifier.evaluate(clothes['dev'])['f'] <= 0.63
        assert 0.87 <= classifier.evaluate(clothes['test'])['f'] <= 0.88

    def test_train_multiclass(self, toxic):
        classifier = SVMClassifier.train(toxic['train'])
        assert 0.99 <= classifier.evaluate(toxic['train'])['f'] <= 1.00
        assert 0.79 <= classifier.evaluate(toxic['dev'])['f'] <= 0.80
        assert 0.83 <= classifier.evaluate(toxic['test'])['f'] <= 0.84

    def test_search_hyperparameters_binary(self, imdb):
        params = SVMClassifier.search_hyperparameters(imdb['train'], imdb['dev'], n_trials=2)
        assert params['min_df'] >= 1
        assert params['max_df'] < 1.0
        assert params['loss'] in ['hinge', 'squared_hinge']
        assert 0.25 <= params['c'] <= 2.0
        assert 500 <= params['max_iter'] <= 1500
        assert params['top_k'] is None
        params = SVMClassifier.search_hyperparameters(imdb['train'], imdb['dev'], n_trials=1, search_top_k=True)
        assert 1 <= params['top_k'] < 2

    def test_search_hyperparameters_multiclass(self, clothes):
        params = SVMClassifier.search_hyperparameters(clothes['train'], clothes['dev'], n_trials=2)
        assert params['min_df'] >= 1
        assert params['max_df'] < 1.0
        assert params['loss'] in ['hinge', 'squared_hinge']
        assert 0.25 <= params['c'] <= 2.0
        assert 500 <= params['max_iter'] <= 1500
        assert params['top_k'] is None
        params = SVMClassifier.search_hyperparameters(clothes['train'], clothes['dev'], n_trials=1, search_top_k=True)
        assert 1 <= params['top_k'] < 5

    def test_search_hyperparameters_multilabel(self, toxic):
        params = SVMClassifier.search_hyperparameters(toxic['train'], toxic['dev'], n_trials=2)
        assert params['min_df'] >= 1
        assert params['max_df'] < 1.0
        assert params['loss'] in ['hinge', 'squared_hinge']
        assert 0.25 <= params['c'] <= 2.0
        assert 500 <= params['max_iter'] <= 1500
        assert params['top_k'] is None
        params = SVMClassifier.search_hyperparameters(toxic['train'], toxic['dev'], n_trials=1, search_top_k=True)
        assert 1 <= params['top_k'] < 6

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

    def test_topk(self, toxic):
        params = SVMClassifier.search_hyperparameters(toxic['train'], toxic['dev'], n_trials=1, search_top_k=True)
        assert 1 <= params['top_k'] < 6
        classifier = SVMClassifier.train(toxic['train'], top_k=3)
