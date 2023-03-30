from .config import Config

from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np

from abc import ABC, abstractmethod
import logging
import os
import json
import pdb

logging.basicConfig(level=logging.INFO)

class Classifier(ABC):
    
    def __init__(self, model, config):
        self._model = model
        self._config = config


    # PUBLIC ABSTRACT methods
    
    @staticmethod
    @classmethod
    def train(cls, train_split, dev_split=None, n_trials=0, f_beta=1, top_k=False, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def predict_probabilities(texts, model):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass

    
    # PUBLIC methods
    
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

    def classify(self, texts, threshold=0.5):
        y_pred = self.predict_probabilities(texts, self._model)
        return y_pred

    def evaluate(self, test_split, beta=1, top_k=None):
        return self._evaluate_model(test_split, self._model, beta, top_k)

    def _load(self, path):
        self._config = Config.load(path)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self._config.save(path)

        
    # PROTECTED ABSTRACT methods

    @staticmethod
    @abstractmethod
    def _default_hyperparameters(top_k=False):
        pass

    @staticmethod
    @abstractmethod
    def _sample_hyperparameters(trial, top_k=False):
        pass    

    
    # PROTECTED methods
        
    @classmethod
    def _evaluate_model(cls, test_split, model, beta=1, top_k=None):
        probas = cls.predict_probabilities(test_split.X, model)
        return cls._evaluate_probabilities(test_split.y, probas, beta, top_k)

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

