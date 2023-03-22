import text_classification as tcls
import general_functions as helper
from dataset import Dataset
from svm_classifier import SVMClassifier
from transformer_classifier import TransformerClassifier

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
        #'dev': args.dev,
        'test': args.test,
    }
    dataset = Dataset.load(split_files, multilabel=False, label_column=args.label)

    #dataset.lemmatize(args.hunspell)
    #dataset['train'].oversample()

    classifier = TransformerClassifier.train(True, dataset['train'], n_trials=args.trials, model_id='orai-nlp/ElhBERTeu')
    metrics = classifier.evaluate(dataset['test'])
    #metrics = TransformerClassifier.cross_validate(False, dataset['train'], model_id='dccuchile/bert-base-spanish-wwm-uncased', n_folds=10)
    print(metrics)
    #classifier.save(args.outdir)
    
    #classifier = TransformerClassifier.load("/tmp/kk_model", is_multilabel=True)
    #classifier.predict_probabilities(["Hau beldurrezko adibide bat da"], classifier._model)
    #print(classifier.classify("Batek eta besteak ere bai"))
    
    #classifier = SVMClassifier.train(True, dataset['train'], dataset['test'], n_trials=args.trials, n_jobs=args.jobs)
            
    #save(estimator, dataset.label_binarizer, f, p, r, args.outdir)
    
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
                        required=False,
                        help="Output path where the model will be created")
    args = parser.parse_args()

    main(args)
