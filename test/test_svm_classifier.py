from classification.svm_classifier import SVMClassifier, SVMConfig

import pytest
import tempfile
import pdb

class TestSVMClassifier:

    def test_train_binary(self, imdb):        
        classifier = SVMClassifier.train(imdb['train'])
        assert 0.99 <= classifier.evaluate(imdb['train']).f('micro') <= 1.00
        assert 0.90 <= classifier.evaluate(imdb['dev']).f('micro') <= 0.91
        assert 0.85 <= classifier.evaluate(imdb['test']).f('micro') <= 0.86

    def test_train_multiclass(self, clothes):
        classifier = SVMClassifier.train(clothes['train'])
        assert 0.94 <= classifier.evaluate(clothes['train']).f('micro') <= 0.95
        assert 0.62 <= classifier.evaluate(clothes['dev']).f('micro') <= 0.63
        assert 0.87 <= classifier.evaluate(clothes['test']).f('micro') <= 0.88

    def test_train_multiclass(self, toxic):
        classifier = SVMClassifier.train(toxic['train'])
        assert 0.99 <= classifier.evaluate(toxic['train']).f('micro') <= 1.00
        assert 0.79 <= classifier.evaluate(toxic['dev']).f('micro') <= 0.80
        assert 0.83 <= classifier.evaluate(toxic['test']).f('micro') <= 0.84

    def test_search_hyperparameters_binary(self, imdb):
        config = SVMClassifier.search_hyperparameters(imdb['train'], imdb['dev'], n_trials=2)
        assert config.min_df >= 1
        assert config.max_df < 1.0
        assert config.loss in ['hinge', 'squared_hinge']
        assert 0.25 <= config.C <= 2.0
        assert 500 <= config.max_iter <= 1500
        assert config.top_k is None
        config = SVMClassifier.search_hyperparameters(imdb['train'], imdb['dev'], n_trials=1, search_top_k=True)
        assert 1 <= config.top_k < 2

    def test_search_hyperparameters_multiclass(self, clothes):
        config = SVMClassifier.search_hyperparameters(clothes['train'], clothes['dev'], n_trials=2)
        assert config.min_df >= 1
        assert config.max_df < 1.0
        assert config.loss in ['hinge', 'squared_hinge']
        assert 0.25 <= config.C <= 2.0
        assert 500 <= config.max_iter <= 1500
        assert config.top_k is None
        config = SVMClassifier.search_hyperparameters(clothes['train'], clothes['dev'], n_trials=1, search_top_k=True)
        assert 1 <= config.top_k < 5

    def test_search_hyperparameters_multilabel(self, toxic):
        config = SVMClassifier.search_hyperparameters(toxic['train'], toxic['dev'], n_trials=2)
        assert config.min_df >= 1
        assert config.max_df < 1.0
        assert config.loss in ['hinge', 'squared_hinge']
        assert 0.25 <= config.C <= 2.0
        assert 500 <= config.max_iter <= 1500
        assert config.top_k is None
        config = SVMClassifier.search_hyperparameters(toxic['train'], toxic['dev'], n_trials=1, search_top_k=True)
        assert 1 <= config.top_k < 6

    def test_train_with_config(self, imdb):
        config = SVMConfig(
            min_df=3,
            max_df=0.8,
            loss='hinge',
            C=1.5,
            max_iter=100,
            top_k=None,
        )
        model = SVMClassifier.train(imdb['train'], config=config)
        assert model.config.type == 'SVM'
        assert model.config.is_multilabel == False
        assert model.config.C == 1.5
        assert model.config.top_k == None
        with pytest.raises(ValueError):
            config.max_df = 2
            SVMClassifier.train(imdb['train'], config=config)

    def test_topk(self, toxic):
        config = SVMClassifier.search_hyperparameters(toxic['train'], toxic['dev'], n_trials=1, search_top_k=True)
        assert 1 <= config.top_k < 6
        classifier = SVMClassifier.train(toxic['train'], config=config)

    def test_save_load(self, imdb, imdb_svm):
        with tempfile.TemporaryDirectory() as path:
            imdb_svm.save(path)
            imdb_svm_ = SVMClassifier.load(path)
        expected_f = imdb_svm.evaluate(imdb['test']).f('micro')
        assert imdb_svm_.evaluate(imdb['test']).f('micro') == expected_f
