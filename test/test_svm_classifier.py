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
        
