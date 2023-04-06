from classification.transformer_classifier import TransformerClassifier

import pytest
import pdb

class TestTransformerClassifier:

    def test_train_binary(self, imdb):
        classifier = TransformerClassifier.train(imdb['train'], model_id='distilbert-base-uncased')
        assert 0.37 <= classifier.evaluate(imdb['dev']).f('weighted') <= 0.39

    def test_train_multiclass(self, clothes):
        classifier = TransformerClassifier.train(clothes['train'], model_id='distilbert-base-uncased')
        assert 0.22 <= classifier.evaluate(clothes['dev']).f('weighted') <= 0.25

    def test_train_multilabel(self, toxic):
        classifier = TransformerClassifier.train(toxic['train'], model_id='distilbert-base-uncased')
        assert 0.33 <= classifier.evaluate(toxic['dev']).f('weighted') <= 0.35
