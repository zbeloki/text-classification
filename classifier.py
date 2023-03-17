from abc import ABC, abstractmethod

from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np

import pdb

class Classifier(ABC):

    def __init__(self):
        self._model = None

    @staticmethod
    def load(path):
        pass

    @abstractmethod
    def train(self, train_split, dev_split=None):
        pass

    @abstractmethod
    def _hyperparameters(self, trial=None):
        pass

    @staticmethod
    @abstractmethod
    def predict_probabilities(texts, model):
        pass

    @classmethod
    def predict(cls, texts, model, threshold=0.5):
        probas = cls.predict_probabilities(texts, model)
        y_pred = np.where(probas > threshold, 1, 0)
        return y_pred

    def classify(self, texts, threshold=0.5):
        y_pred = self.predict(texts, self._model)
        return y_pred

    def evaluate(self, test_split, beta=1):
        return self._evaluate_model(test_split, self._model, beta)
        
    def save(self, path):
        pass

    @classmethod
    def _evaluate_model(cls, test_split, model, beta=1):
        y_pred = cls.predict(test_split.X, model)
        f = fbeta_score(test_split.y, y_pred, beta=beta, average='micro')
        p = precision_score(test_split.y, y_pred, average='micro')
        r = recall_score(test_split.y, y_pred, average='micro')
        metrics = {'f':f, 'p':p, 'r':r}
        return metrics
