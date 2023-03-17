from classifier import Classifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

class SVMClassifier(Classifier):

    def __init__(self):
        super().__init__()

    @staticmethod
    def load(path):
        pass

    @staticmethod
    def predict_probabilities(texts, estimator):
        probas = estimator.predict_proba(texts)
        return probas

    def train(self, train_split, dev_split=None, n_trials=0, n_jobs=1):

        params = self._hyperparameters()
        model, metrics = self.training_trial(params, train_split, dev_split, n_jobs)

        if n_trials > 0:
            def objective(trial):
                params = self._hyperparameters(trial)
                metrics = self.training_trial(params, train_split, dev_split, n_jobs)
                return metrics['f']
            warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            opt_params = study.best_params
            opt_model, opt_metrics = self.training_trial(opt_params, train_split, dev_split, n_jobs)

            if opt_metrics['f'] > metrics['f']:
                metrics, params, model = opt_metrics, opt_params, opt_model

        self._model = model

    def training_trial(self, params, train_split, dev_split=None, n_jobs=1):

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
        metrics = self._evaluate_model(dev_split, pipe)

        return pipe, metrics

    def _hyperparameters(self, trial=None):

        params = {}
        if trial is None:
            params['min_df'] = 1
            params['max_df'] = 1.0
            params['loss'] = 'squared_hinge'
            params['c'] = 1.0
            params['max_iter'] = 1000
        else:
            params['min_df'] = trial.suggest_int("min_df", 1, 100, log=True)
            params['max_df'] = trial.suggest_float("max_df", 0.25, 1.0)
            params['loss'] = trial.suggest_categorical("loss", ['hinge', 'squared_hinge'])
            params['c'] = trial.suggest_float("c", 0.25, 2.0)
            params['max_iter'] = trial.suggest_int("max_iter", 500, 1500, log=True)

        return params

    def save(self, path):
        pass
