from classification.dataset import Dataset, DatasetSplit
from classification.svm_classifier import SVMClassifier

import pandas as pd

import pytest
import pathlib

import os

@pytest.fixture(scope='session')
def datadir():
    return pathlib.Path(__file__).parent.resolve() / 'data'

@pytest.fixture(scope='session')
def hunspell(datadir):
    return os.path.join(datadir, 'en_US')

# Datasets

@pytest.fixture(scope='session')
def imdb(datadir):
    return Dataset.load({
        'train': datadir/"imdb.train.tsv",
        'dev': datadir/"imdb.dev.tsv",
        'test': datadir/"imdb.test.tsv",
    }, label_column='sentiment')

@pytest.fixture(scope='function')
def imdb_func(datadir):
    return Dataset.load({
        'train': datadir/"imdb.train.tsv",
        'dev': datadir/"imdb.dev.tsv",
        'test': datadir/"imdb.test.tsv",
    }, label_column='sentiment')

@pytest.fixture(scope='session')
def imdb_lem(datadir):
    return Dataset.load({
        'train': datadir/"imdb.train.lem.tsv",
        'dev': datadir/"imdb.dev.lem.tsv",
        'test': datadir/"imdb.test.lem.tsv",
    }, label_column='sentiment')

@pytest.fixture(scope='session')
def imdb_test(imdb):
    return imdb['test']

@pytest.fixture(scope='session')
def clothes(datadir):
    return Dataset.load({
        'train': datadir/"clothes.train.tsv",
        'dev': datadir/"clothes.dev.tsv",
        'test': datadir/"clothes.test.tsv",
    })

@pytest.fixture(scope='session')
def clothes_test(clothes):
    return clothes['test']

@pytest.fixture(scope='session')
def toxic(datadir):
    return Dataset.load({
        'train': datadir/"toxic.train.csv",
        'dev': datadir/"toxic.dev.csv",
        'test': datadir/"toxic.test.csv",
    }, label_column='tags', label_sep=',')

@pytest.fixture(scope='session')
def toxic_test(toxic):
    return toxic['test']

@pytest.fixture(scope='function')
def small_mc():
    return DatasetSplit(pd.DataFrame({
        'id': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'text': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
        'labels': ['A', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C'],
        # A:1, B:2, C:6
    }))

@pytest.fixture(scope='function')
def small_ml():
    return DatasetSplit(pd.DataFrame({
        'id': ['1', '2', '3', '4', '5', '6', '7'],
        'text': ['one', 'two', 'three', 'four', 'five', 'six', 'seven'],
        'labels': [['A', 'B'], ['B'], ['B', 'A', 'C'], ['A'], ['A', 'B'], ['B'], ['C']],
        # A:4, B:5, C:2
    }))

# models

@pytest.fixture(scope='session')
def imdb_svm(imdb):
    return SVMClassifier.train(imdb['train'], min_df=1, max_df=1.0, loss='squared_hinge', c=1.0, max_iter=1000)

@pytest.fixture(scope='session')
def clothes_svm(clothes):
    return SVMClassifier.train(clothes['train'], min_df=1, max_df=1.0, loss='squared_hinge', c=1.0, max_iter=1000)

@pytest.fixture(scope='session')
def toxic_svm(toxic):
    return SVMClassifier.train(toxic['train'], min_df=1, max_df=1.0, loss='squared_hinge', c=1.0, max_iter=1000)

# classification outputs

@pytest.fixture(scope='session')
def imdb_svm_out(imdb, imdb_svm):
    return imdb_svm.classify(imdb['dev'].X)

@pytest.fixture(scope='session')
def clothes_svm_out(clothes, clothes_svm):
    return clothes_svm.classify(clothes['dev'].X)

@pytest.fixture(scope='session')
def toxic_svm_out(toxic, toxic_svm):
    return toxic_svm.classify(toxic['dev'].X)