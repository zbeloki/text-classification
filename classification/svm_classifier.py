from .classifier import Classifier
from .config import Config

from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning

import optuna

import joblib
import json
import os
import pdb
import warnings

class SVMClassifier(Classifier):

    def __init__(self, model, config):
        super().__init__(model, config)

    @classmethod
    def train(cls, train_split, dev_split=None, n_trials=0, f_beta=1, top_k=False, **kwargs):
        n_jobs = kwargs['n_jobs']
        config = Config.from_dict(kwargs)

        params = cls._default_hyperparameters(top_k)
        model, metrics = cls._training_trial(params, train_split, dev_split, n_jobs, f_beta)
        print(f"Default hyperparameters: {metrics}")

        if n_trials > 0:
            
            def objective(trial):
                params = cls._sample_hyperparameters(trial, top_k)
                _, metrics = cls._training_trial(params, train_split, dev_split, n_jobs, f_beta)
                return metrics['f']

            warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            params = study.best_params
            model, metrics = cls._training_trial(params, train_split, dev_split, n_jobs, f_beta)

        return SVMClassifier(model, config)

    @classmethod
    def _training_trial(cls, params, train_split, dev_split=None, n_jobs=1, f_beta=1):

        tfidf_vectorizer = TfidfVectorizer(min_df=params['min_df'], max_df=params['max_df'])
        estimator = LinearSVC(loss=params['loss'], max_iter=params['max_iter'], C=params['c'])
        estimator = CalibratedClassifierCV(estimator)
        estimator = OneVsRestClassifier(estimator, n_jobs=n_jobs)

        pipe = Pipeline([
            ('tfidf', tfidf_vectorizer),
            ('model', estimator),
        ])
        pipe.fit(train_split.X, train_split.y)

        if dev_split is None:
            dev_split = train_split
        metrics = cls._evaluate_model(dev_split, pipe, f_beta, params['top_k'])

        return pipe, metrics

    @staticmethod
    def predict_probabilities(texts, estimator):
        probas = estimator.predict_proba(texts)
        return probas

    @staticmethod
    def _default_hyperparameters(top_k=False):
        params = {
            'min_df': 1,
            'max_df': 1.0,
            'loss': 'squared_hinge',
            'c': 1.0,
            'max_iter': 1000,
            'top_k': 1 if top_k else None,
        }
        return params

    @staticmethod
    def _sample_hyperparameters(trial, top_k):
        params = {
            'min_df': trial.suggest_int("min_df", 1, 100, log=True),
            'max_df': trial.suggest_float("max_df", 0.25, 1.0),
            'loss': trial.suggest_categorical("loss", ['hinge', 'squared_hinge']),
            'c': trial.suggest_float("c", 0.25, 2.0),
            'max_iter': trial.suggest_int("max_iter", 500, 1500, log=True),
            'top_k': trial.suggest_int("top_k", 1, 10) if top_k else None,
        }
        return params

    @classmethod
    def load(cls, path):
        classifier = SVMClassifier(None, None, None)
        classifier._load(path)
        return classifier
    
    def _load(self, path):
        super().load(path)
        fname = "svm.joblib"
        self._model = joblib.load(os.path.join(path, fname))

    def save(self, path):
        super().save(path)
        fpath = os.path.join(path, "svm.joblib")
        joblib.dump(self._model, fpath)
