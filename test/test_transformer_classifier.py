from classification.transformer_classifier import TransformerClassifier, TransformerConfig

import pytest
import tempfile
import pdb

@pytest.fixture(scope='class')
def imdb_transformer(imdb):
    config = TransformerConfig('distilbert-base-uncased')
    return TransformerClassifier.train(imdb['train'], config=config)

class TestTransformerClassifier:

    def test_train_binary(self, imdb, imdb_transformer):
        assert 0.75 <= imdb_transformer.evaluate(imdb['dev']).f('weighted') <= 0.85

    def test_train_multiclass(self, clothes):
        config = TransformerConfig('distilbert-base-uncased')
        classifier = TransformerClassifier.train(clothes['train'], config=config)
        assert 0.70 <= classifier.evaluate(clothes['dev']).f('weighted') <= 0.75

    def test_train_multilabel(self, toxic):
        config = TransformerConfig('distilbert-base-uncased')
        classifier = TransformerClassifier.train(toxic['train'], config=config)
        assert 0.65 <= classifier.evaluate(toxic['dev']).f('weighted') <= 0.70

    def test_save_load(self, imdb, imdb_transformer):
        with tempfile.TemporaryDirectory() as path:
            imdb_transformer.save(path)
            imdb_transformer_ = TransformerClassifier.load(path)
        expected_f = imdb_transformer.evaluate(imdb['test']).f('micro')
        assert imdb_transformer_.evaluate(imdb['test']).f('micro') == expected_f
