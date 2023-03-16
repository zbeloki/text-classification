import text_classification as tcls
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
import joblib
import json
import os
import pdb

def main(train_fpath, dev_fpath, test_fpath, label_field, cv_kfolds, hunspell_path, outdir):
    
    data = {}
    data['train'] = tcls.load_dataset_split(train_fpath, 'labels')
    if dev_fpath is not None: data['dev'] = tcls.load_dataset_split(dev_fpath, 'labels')
    if test_fpath is not None: data['test'] = tcls.load_dataset_split(test_fpath, 'labels')

    label_binarizer = tcls.binarize_labels(data, multilabel=True)
    #tcls.lemmatize(data['train'], hunspell_path)
    
    data['train_os'] = tcls.oversample(data['train'])

    estimator = tcls.train_svm(data['train'], n_jobs=16)
    
    text_classifier = tcls.TextClassifier(estimator, label_binarizer)
    f, p, r = tcls.evaluate_split(data['dev'], text_classifier, beta=1)

    pdb.set_trace()
    
    

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
    parser.add_argument("--label-field",
                        required=False,
                        help="Which column to use as labels. By default, the first column except 'id' and 'text' is used.")
    parser.add_argument("--cv_kfolds",
                        required=False,
                        type=int,
                        help="If given, Cross-Validation with this number of folds will be used on data passed in argument 'train', and 'dev' and 'test' sets will be ignored")
    parser.add_argument("--hunspell",
                        required=False,
                        help="Hunspell hiztegiaren patha, luzapenik gabe")
    parser.add_argument("--outdir",
                        required=True,
                        help="Output path where the model will be created")
    args = parser.parse_args()

    main(args.train, args.dev, args.test, args.label_field, args.cv_kfolds, args.hunspell, args.outdir)
