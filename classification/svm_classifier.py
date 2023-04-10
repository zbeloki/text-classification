from .classifier import Classifier
from .config import Config

from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning

import optuna
import numpy as np

from typing import Literal, Union
import joblib
import json
import os
import pdb
import warnings

class SVMConfig(Config):

    def __init__(self,
                 min_df=None,
                 max_df=None,
                 loss=None,
                 C=None,
                 max_iter=None,
                 n_jobs=None,
                 is_multilabel=None,
                 f_beta=1.0,
                 optim_avg='weighted',
                 top_k=None):
        super().__init__(is_multilabel, f_beta, optim_avg, top_k)
        self.min_df = min_df
        self.max_df = max_df
        self.loss = loss
        self.C = C
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    @classmethod
    def from_args(cls, args):
        pass

    
class SVMClassifier(Classifier):

    def __init__(self, config, label_binarizer, model):
        super().__init__(config, label_binarizer)
        self._model = model

    @classmethod
    def train(cls, train_split, dev_split=None, config=None):
        classifier, metrics = cls._training_trial(train_split, dev_split, config)
        print(f"Default hyperparameters: {metrics}")
        return classifier

    @classmethod
    def search_hyperparameters(cls, train_split, dev_split, n_trials=10, f_beta=1, search_top_k=False, n_jobs=1, hp_space=None, **kwargs):

        def objective(trial):
            if hp_space is not None:
                config = hp_space()
            else:
                config = SVMConfig()
                config.min_df = trial.suggest_int("min_df", 1, 100, log=True)
                config.max_df = trial.suggest_float("max_df", 0.25, 1.0)
                config.loss = trial.suggest_categorical("loss", ['hinge', 'squared_hinge'])
                config.C = trial.suggest_float("C", 0.25, 2.0)
                config.max_iter = trial.suggest_int("max_iter", 500, 1500, log=True)
                if search_top_k:
                    max_top_k = min(10, train_split.n_classes-1)
                    config.top_k = trial.suggest_int("top_k", 1, max_top_k) if search_top_k else None
            _, metrics = cls._training_trial(train_split, dev_split, config)
            return metrics.f(average=config.optim_avg)

        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        study = optuna.create_study(direction="maximize")
        # use 'catch' to ignore trials where min_df > max_df
        study.optimize(objective, n_trials=n_trials, catch=(ValueError))
        
        return SVMConfig.from_dict(study.best_params)

    @classmethod
    def _training_trial(cls, train_split, dev_split, config):

        if config is None:
            config = SVMConfig()
        config.type = 'SVM'
        if config.is_multilabel is None:
            config.is_multilabel = train_split.is_multilabel

        tfidf_vectorizer = TfidfVectorizer(**config.kwargs(['min_df', 'max_df']))
        estimator = LinearSVC(**config.kwargs(['loss', 'max_iter', 'C']))
        estimator = CalibratedClassifierCV(estimator)
        estimator = OneVsRestClassifier(estimator, **config.kwargs(['n_jobs']))

        pipe = Pipeline([
            ('tfidf', tfidf_vectorizer),
            ('model', estimator),
        ])

        X = train_split.X
        label_binarizer = train_split.create_label_binarizer()
        y = train_split.y(label_binarizer)
        ohv = train_split.ohv(label_binarizer)
        if min(np.sum(ohv, axis=0)) < 2:
                raise ValueError("CalibratedClassifierCV needs at least 2 examples from each class")

        pipe.fit(X, y)

        if dev_split is None:
            dev_split = train_split
        classifier = SVMClassifier(config, label_binarizer, pipe)
        metrics = classifier.evaluate(dev_split, **config.kwargs(['f_beta', 'top_k']))

        return classifier, metrics

    def predict_probabilities(self, texts):
        y_proba = self._model.predict_proba(texts)
        return y_proba

    @classmethod
    def load(cls, path):
        kwargs = cls._load(path)
        fname = "svm.joblib"
        model = joblib.load(os.path.join(path, fname))
        return SVMClassifier(model=model, **kwargs)
    
    def save(self, path):
        super().save(path)
        fpath = os.path.join(path, "svm.joblib")
        joblib.dump(self._model, fpath)


