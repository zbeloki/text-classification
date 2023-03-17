import text_classification as tcls
import general_functions as helper
from data import Dataset

from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import numpy as np
import optuna

import argparse
import logging
import joblib
import json
import os
import pdb

logging.basicConfig(level=logging.INFO)

def main(args):

    split_files = {
        'train': args.train,
        'test': args.test,
    }
    dataset = tcls.Dataset.load(split_files, multilabel=True, label_column=args.label)

    #dataset.lemmatize(args.hunspell)
    #dataset['train'].oversample()

    estimator = train_svm(dataset['train'], n_jobs=args.jobs)
    f, p, r = tcls.evaluate(estimator, dataset['test'].X, dataset['test'].y)
    logging.info(f"Default config: F:{f:.3f}, P:{p:.3f}, R:{r:.3f}")

    if args.trials > 0:
        params = helper.optimize_hyperparameters(objective, dataset['train'], dataset['test'], n_trials=args.trials, n_jobs=args.jobs)
        logging.info(f"Best parameters: {params}")
    
        estimator_opt = train_svm(dataset['train'], n_jobs=args.jobs, **params)
        f_opt, p_opt, r_opt = tcls.evaluate(estimator_opt, dataset['test'].X, dataset['test'].y)
        logging.info(f"Optimized config: F:{f:.3f}, P:{p:.3f}, R:{r:.3f}")

        if f_opt > f:
            estimator = estimator_opt
            f, p, r = f_opt, p_opt, r_opt
            
    save(estimator, dataset.label_binarizer, f, p, r, args.outdir)
    
    
def train_svm(train, min_df=1, max_df=1.0, loss_f='squared_hinge', c=1.0, max_iter=1000, n_jobs=1):

    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    estimator = LinearSVC(loss=loss_f, max_iter=max_iter, C=c)
    estimator = CalibratedClassifierCV(estimator)
    estimator = OneVsRestClassifier(estimator, n_jobs=n_jobs)

    pipe = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('model', estimator),
    ])
    pipe.fit(train.X, train.y)

    return pipe

def objective(trial, train, dev, n_jobs):

    min_df = trial.suggest_int("min_df", 1, 100, log=True)
    max_df = trial.suggest_float("max_df", 0.25, 1.0)
    loss_f = trial.suggest_categorical("loss_f", ['hinge', 'squared_hinge'])
    c = trial.suggest_float("c", 0.25, 2.0)
    max_iter = trial.suggest_int("max_iter", 500, 1500, log=True)
    
    estimator = train_svm(train, min_df, max_df, loss_f, c, max_iter, n_jobs=n_jobs)
    f, _, _ = tcls.evaluate(estimator, dev.X, dev.y)
    
    return f

def save(estimator, label_binarizer, f, p, r, path):

    if not os.path.exists(path):
        os.makedirs(path)
        
    fpath = os.path.join(path, 'model.joblib')
    joblib.dump(estimator, fpath)
    fpath = os.path.join(path, 'label_binarizer.joblib')
    joblib.dump(label_binarizer, fpath)

    # dump config
    config = {}
    name = type(estimator).__name__
    config[name] = {}
    for param_key, param_val in estimator.get_params().items():
        try:
            json.dumps(param_val)
            config[name][param_key] = param_val
        except TypeError:
            config[name][param_key] = type(param_val).__name__
    fpath = os.path.join(path, 'config.json')
    with open(fpath, 'w') as f:
        json.dump(config, f, indent=4)

    # dump evaluation reports                                                                                                                                                                
    fpath = os.path.join(path, 'results_test.txt')
    with open(fpath, 'w') as fo:
        print(f"F: {f}", file=fo)
        print(f"P: {p}", file=fo)
        print(f"R: {r}", file=fo)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train",
                        required=True,
                        help="Train dataset TSV (id, text, labels1, [labels2, ...])")
    parser.add_argument("--dev",
                        required=False,
                        help="Dev dataset TSV (id, text, labels1, [labels2, ...])")
    parser.add_argument("--test",
                        required=False,
                        help="Test dataset TSV (id, text, labels1, [labels2, ...])")
    parser.add_argument("--label",
                        default='labels',
                        help="Which column to use as labels. By default, the first column except 'id' and 'text' is used.")
    parser.add_argument("--cv_kfolds",
                        required=False,
                        type=int,
                        help="If given, Cross-Validation with this number of folds will be used on data passed in argument 'train', and 'dev' and 'test' sets will be ignored")
    parser.add_argument("--trials",
                        type=int,
                        default=0,
                        help="How many trials to run by Optuna for hyperparameter searching")
    parser.add_argument("--jobs",
                        type=int,
                        default=1,
                        help="Number of parallel jobs to train the model")
    parser.add_argument("--hunspell",
                        required=False,
                        help="Hunspell hiztegiaren patha, luzapenik gabe")
    parser.add_argument("--outdir",
                        required=True,
                        help="Output path where the model will be created")
    args = parser.parse_args()

    main(args)
