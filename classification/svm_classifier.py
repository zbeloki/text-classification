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

import joblib
import json
import os
import pdb
import warnings

class SVMClassifier(Classifier):

    def __init__(self, config, label_binarizer, model):
        super().__init__(config, label_binarizer)
        self._model = model

    @classmethod
    def train(cls, train_split, dev_split=None, f_beta=1, top_k=None, n_jobs=1, min_df=1, max_df=1.0, loss='squared_hinge', c=1.0, max_iter=1000, **kwargs):
        config = Config.from_dict(kwargs)
        classifier, metrics = cls._training_trial(train_split, dev_split, config, n_jobs, f_beta, min_df, max_df, loss, c, max_iter, top_k)
        print(f"Default hyperparameters: {metrics}")
        return classifier

    @classmethod
    def search_hyperparameters(cls, train_split, dev_split, n_trials=10, f_beta=1, search_top_k=False, n_jobs=1, **kwargs):
        config = Config.from_dict(kwargs)
            
        def objective(trial):
            min_df = trial.suggest_int("min_df", 1, 100, log=True)
            max_df = trial.suggest_float("max_df", 0.25, 1.0)
            loss = trial.suggest_categorical("loss", ['hinge', 'squared_hinge'])
            c = trial.suggest_float("c", 0.25, 2.0)
            max_iter = trial.suggest_int("max_iter", 500, 1500, log=True)
            if search_top_k:
                max_top_k = min(10, train_split.n_classes-1)
                top_k = trial.suggest_int("top_k", 1, max_top_k) if search_top_k else None
            else:
                top_k = trial.suggest_categorical('top_k', [None])
            _, metrics = cls._training_trial(train_split, dev_split, config, n_jobs, f_beta, min_df, max_df, loss, c, max_iter, top_k)
            return metrics['f']

        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        study = optuna.create_study(direction="maximize")
        # use 'catch' to ignore trials where min_df > max_df
        study.optimize(objective, n_trials=n_trials, catch=(ValueError))

        return study.best_params

    @classmethod
    def _training_trial(cls, train_split, dev_split, config, n_jobs, f_beta, min_df, max_df, loss, c, max_iter, top_k):

        tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
        estimator = LinearSVC(loss=loss, max_iter=max_iter, C=c)
        estimator = CalibratedClassifierCV(estimator)
        estimator = OneVsRestClassifier(estimator, n_jobs=n_jobs)

        pipe = Pipeline([
            ('tfidf', tfidf_vectorizer),
            ('model', estimator),
        ])

        label_binarizer = train_split.create_label_binarizer()
        X, y = train_split.X, train_split.y(label_binarizer)
        if min(np.sum(y, axis=0)) < 2:
            raise ValueError("CalibratedClassifierCV needs at least 2 examples from each class")

        pipe.fit(X, y)

        if dev_split is None:
            dev_split = train_split
        classifier = SVMClassifier(config, label_binarizer, pipe)
        metrics = classifier.evaluate(dev_split, f_beta, top_k)

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
