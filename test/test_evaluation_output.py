import pytest

from classification.evaluation_output import EvaluationOutput
import numpy as np

import pdb

@pytest.fixture(scope='class')
def imdb_eval(imdb, imdb_svm):
    y_true = imdb['test'].y(imdb_svm._label_binarizer)
    y_proba = imdb_svm.predict_probabilities(imdb['test'].X)
    return EvaluationOutput(y_true, y_proba, False, label_binarizer=imdb_svm._label_binarizer)

@pytest.fixture(scope='class')
def clothes_eval(clothes, clothes_svm):
    y_true = clothes['test'].y(clothes_svm._label_binarizer)
    y_proba = clothes_svm.predict_probabilities(clothes['test'].X)
    return EvaluationOutput(y_true, y_proba, False, label_binarizer=clothes_svm._label_binarizer)

@pytest.fixture(scope='class')
def toxic_eval(toxic, toxic_svm):
    y_true = toxic['test'].y(toxic_svm._label_binarizer)
    y_proba = toxic_svm.predict_probabilities(toxic['test'].X)
    return EvaluationOutput(y_true, y_proba, True, label_binarizer=toxic_svm._label_binarizer)

@pytest.fixture(scope='class')
def toxic_eval_topk(toxic, toxic_svm):
    y_true = toxic['test'].y(toxic_svm._label_binarizer)
    y_proba = toxic_svm.predict_probabilities(toxic['test'].X)
    return EvaluationOutput(y_true, y_proba, True, beta=2.0, threshold=0.25, top_k=3)

class TestEvaluationOutput:

    def test_accuracy(self, imdb_eval, clothes_eval):
        assert np.around(imdb_eval.accuracy(), decimals=3) == 0.850
        assert np.around(clothes_eval.accuracy(), decimals=3) == 0.860

    def test_f(self, imdb_eval, clothes_eval, toxic_eval, toxic_eval_topk):
        assert np.around(imdb_eval.f('micro'), decimals=3) == 0.850
        assert np.around(imdb_eval.f('macro'), decimals=3) == 0.840
        assert np.around(imdb_eval.f('weighted'), decimals=3) == 0.848
        assert np.around(imdb_eval.f('binary'), decimals=3) == 0.880
        assert np.around(clothes_eval.f('micro'), decimals=3) == 0.860
        assert np.around(clothes_eval.f('macro'), decimals=3) == 0.525
        assert np.around(clothes_eval.f('weighted'), decimals=3) == 0.834
        assert np.around(toxic_eval.f('micro'), decimals=3) == 0.831
        assert np.around(toxic_eval.f('macro'), decimals=3) == 0.691
        assert np.around(toxic_eval.f('weighted'), decimals=3) == 0.833
        assert np.around(toxic_eval_topk.f('weighted'), decimals=3) == 0.707

    def test_precision(self, imdb_eval, clothes_eval, toxic_eval, toxic_eval_topk):
        assert np.around(imdb_eval.precision('micro'), decimals=3) == 0.850
        assert np.around(imdb_eval.precision('macro'), decimals=3) == 0.852
        assert np.around(imdb_eval.precision('weighted'), decimals=3) == 0.851
        assert np.around(imdb_eval.precision('binary'), decimals=3) == 0.846
        assert np.around(clothes_eval.precision('micro'), decimals=3) == 0.860
        assert np.around(clothes_eval.precision('macro'), decimals=3) == 0.514
        assert np.around(clothes_eval.precision('weighted'), decimals=3) == 0.817
        assert np.around(toxic_eval.precision('micro'), decimals=3) == 0.889
        assert np.around(toxic_eval.precision('macro'), decimals=3) == 0.726
        assert np.around(toxic_eval.precision('weighted'), decimals=3) == 0.91
        assert np.around(toxic_eval_topk.precision('weighted'), decimals=3) == 0.786

    def test_recall(self, imdb_eval, clothes_eval, toxic_eval, toxic_eval_topk):
        assert np.around(imdb_eval.recall('micro'), decimals=3) == 0.850
        assert np.around(imdb_eval.recall('macro'), decimals=3) == 0.833
        assert np.around(imdb_eval.recall('weighted'), decimals=3) == 0.850
        assert np.around(imdb_eval.recall('binary'), decimals=3) == 0.917
        assert np.around(clothes_eval.recall('micro'), decimals=3) == 0.860
        assert np.around(clothes_eval.recall('macro'), decimals=3) == 0.542
        assert np.around(clothes_eval.recall('weighted'), decimals=3) == 0.860
        assert np.around(toxic_eval.recall('micro'), decimals=3) == 0.780
        assert np.around(toxic_eval.recall('macro'), decimals=3) == 0.676
        assert np.around(toxic_eval.recall('weighted'), decimals=3) == 0.780
        assert np.around(toxic_eval_topk.recall('weighted'), decimals=3) == 0.707
