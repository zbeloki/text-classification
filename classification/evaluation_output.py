from classification.classification_output import ClassificationOutput

from functools import cached_property
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score

import pdb

class EvaluationOutput:

    def __init__(self, y_true, y_proba, is_multilabel, label_binarizer=None, beta=1.0, threshold=None, top_k=None):
        self._y_true = y_true
        self._y_proba = y_proba
        self._beta = beta
        self._threshold = threshold
        if threshold is None:
            self._threshold = 0.5 if top_k is None else 0.0
        self._top_k = top_k
        self._clout = ClassificationOutput(y_proba, is_multilabel, label_binarizer, threshold=threshold, top_k=top_k)

    def accuracy(self):
        return accuracy_score(self._y_true, self._clout.y)

    def f(self, average):
        return fbeta_score(self._y_true, self._clout.y, beta=self._beta, average=average, zero_division=0)

    def precision(self, average):
        return precision_score(self._y_true, self._clout.y, average=average, zero_division=0)

    def recall(self, average):
        return recall_score(self._y_true, self._clout.y, average=average, zero_division=0)
    
