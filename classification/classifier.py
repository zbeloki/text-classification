from .config import Config
from .classification_output import ClassificationOutput

from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np

from abc import ABC, abstractmethod
import logging
import os
import json
import pdb


class Classifier(ABC):
    
    def __init__(self, config, label_binarizer):
        self._config = config
        self._label_binarizer = label_binarizer

    # ABSTRACT methods
    
    @staticmethod
    @classmethod
    def train(cls, train_split, dev_split=None, f_beta=1, top_k=False, *args, **kwargs):
        pass

    @staticmethod
    @classmethod
    def search_hyperparameters(cls, train_split, dev_split, n_trials, f_beta=1, top_k=False):
        pass
    
    @abstractmethod
    def predict_probabilities(self, texts):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass

    # methods
    
    @classmethod
    def cross_validate(cls, dataset, n_folds=5, beta=1, *args, **kwargs):
        all_metrics = []
        for i, (train_split, test_split) in dataset.kfold(n_folds):
            logging.info(f"Training fold {i+1}/{n_folds}")
            classifier = cls.train(train_split, n_trials=0, *args, **kwargs)
            metrics = classifier.evaluate(test_split, beta)
            logging.info(f"Fold {i+1} metrics: {metrics}")
            all_metrics.append(metrics)
        metric_names = all_metrics[0].keys()
        metrics = { m: np.average([ ms[m] for ms in all_metrics ]) for m in metric_names }
        return metrics

    def classify(self, texts, threshold=0.5, top_k=None):
        y_proba = self.predict_probabilities(texts)
        is_multilabel = self.config.classification_type == 'multilabel'
        return ClassificationOutput(y_proba, self._label_binarizer, is_multilabel, threshold, top_k)

    def evaluate(self, test_split, beta=1, top_k=None):
        X, y = test_split.X, test_split.y(self._label_binarizer)
        y_proba = self.predict_probabilities(X)
        return self._evaluate_probabilities(y, y_proba, beta, top_k)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self._config.save(path)
        # save: self._label_binarizer

    @property
    def config(self):
        return self._config

    def _load(self, path):
        return {
            'config': Config.load(path),
            # 'label_binarizer': ...,
        }

    @classmethod
    def _evaluate_logits(cls, y_true, logits, is_multilabel, beta=1, top_k=None):
        if is_multilabel:
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            probas = sigmoid(logits)
            return cls._evaluate_probabilities(y_true, probas, beta, top_k)
        else:
            y_pred = np.argmax(logits, axis=1)
            y_true = np.argmax(y_true, axis=1)
            return cls._evaluate_preds(y_true, y_pred, beta, top_k)

    @classmethod
    def _evaluate_probabilities(cls, y_true, y_proba, beta=1, top_k=None):
        threshold = 0.5 if top_k is None else 0.0
        
        y_proba[y_proba < threshold] = 0
        if top_k is not None:
            threshold_probas = -np.sort(-y_proba)[:, top_k]
            threshold_probas = threshold_probas[..., np.newaxis]
            y_proba[y_proba <= threshold_probas] = 0
        y_proba[y_proba > 0] = 1
        y_pred = y_proba.astype(int)
            
        return cls._evaluate_preds(y_true, y_pred, beta)

    @staticmethod
    def _evaluate_preds(y_true, y_pred, beta=1):
        f = fbeta_score(y_true, y_pred, beta=beta, average='micro')
        p = precision_score(y_true, y_pred, average='micro')
        r = recall_score(y_true, y_pred, average='micro')
        metrics = {'f':f, 'p':p, 'r':r}
        return metrics

