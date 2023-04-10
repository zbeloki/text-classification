from classification.transformer_classifier import TransformerClassifier, TransformerConfig

import pytest
import pdb

class TestTransformerClassifier:

    def test_train_binary(self, imdb):
        config = TransformerConfig('distilbert-base-uncased')
        classifier = TransformerClassifier.train(imdb['train'], config=config)
        assert 0.75 <= classifier.evaluate(imdb['dev']).f('weighted') <= 0.85

    def test_train_multiclass(self, clothes):
        config = TransformerConfig('distilbert-base-uncased')
        classifier = TransformerClassifier.train(clothes['train'], config=config)
        assert 0.70 <= classifier.evaluate(clothes['dev']).f('weighted') <= 0.75

    def test_train_multilabel(self, toxic):
        config = TransformerConfig('distilbert-base-uncased')
        classifier = TransformerClassifier.train(toxic['train'], config=config)
        assert 0.65 <= classifier.evaluate(toxic['dev']).f('weighted') <= 0.70
