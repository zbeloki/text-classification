from .config import Config
from .classification_output import ClassificationOutput
from .evaluation_output import EvaluationOutput

from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np

from abc import ABC, abstractmethod
import logging
import joblib
import os
import json
import pdb


class Classifier(ABC):
    label_binarizer_fname = 'label_binarizer.joblib'
    
    def __init__(self, config, label_binarizer):
        self._config = config
        self._label_binarizer = label_binarizer

    # ABSTRACT methods
    
    @classmethod
    @abstractmethod
    def train(cls, train_split, dev_split=None, config=None):
        pass

    @classmethod
    @abstractmethod
    def search_hyperparameters(cls, train_split, dev_split, n_trials, f_beta=1, search_top_k=False, hp_space=None, **kwargs):
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

    def classify(self, texts, threshold=None, top_k=None):
        y_proba = self.predict_probabilities(texts)
        is_multilabel = self.config.is_multilabel
        return ClassificationOutput(y_proba, is_multilabel, self._label_binarizer, threshold, top_k)

    def evaluate(self, test_split, f_beta=1, threshold=None, top_k=None):
        X, y = test_split.X, test_split.y(self._label_binarizer)
        y_proba = self.predict_probabilities(X)
        is_multilabel = self.config.is_multilabel
        return EvaluationOutput(y, y_proba, is_multilabel, self._label_binarizer, f_beta, threshold, top_k)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self._config.save(path)
        joblib.dump(self._label_binarizer, os.path.join(path, self.label_binarizer_fname))

    @property
    def config(self):
        return self._config

    @classmethod
    def _load(cls, path):
        config = Config.load(path)
        label_binarizer = joblib.load(os.path.join(path, cls.label_binarizer_fname))
        return config, label_binarizer

    @classmethod
    def _evaluate_logits(cls, y_true, logits, is_multilabel, beta=1.0, top_k=None):
        if is_multilabel:
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            probas = sigmoid(logits)
        else:
            pass
            #probas = np.argmax(logits, axis=1) # softmax -> probas -> EvaluationOutput()
        return EvaluationOutput(y_true, probas, beta, top_k)
