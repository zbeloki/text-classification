from classification.dataset import Dataset, DatasetSplit

import numpy as np

import pytest
import pathlib
import os
import pdb

@pytest.fixture(scope='session')
def datadir():
    return pathlib.Path(__file__).parent.resolve() / 'data'

@pytest.fixture(scope='class')
def imdb_train(datadir):
    return DatasetSplit.load(datadir/"imdb.train.tsv")

@pytest.fixture(scope='class')
def imdb_test(datadir):
    return DatasetSplit.load(datadir/"imdb.test.tsv")

@pytest.fixture(scope='class')
def imdb_test_lem(datadir):
    ds = DatasetSplit.load(datadir/"imdb.test.tsv")
    hs_dic = os.path.join(datadir, 'en_US')
    ds.lemmatize(hs_dic)
    return ds

@pytest.fixture(scope='class')
def reuters_train(datadir):
    return DatasetSplit.load(datadir/"reuters.train.csv", label_column='topics', label_sep='|')

@pytest.fixture(scope='class')
def reuters_test(datadir):
    return DatasetSplit.load(datadir/"reuters.test.csv", label_column='topics', label_sep='|')


class TestDatasetSplit:
    
    def test_ids(self, imdb_test, reuters_test):
        # imdb (multiclass)
        assert type(imdb_test.ids) == list
        assert len(imdb_test.ids) == 1000
        assert type(imdb_test.ids[0]) == str
        assert imdb_test.ids[0] == 'r30773'
        # reuters (multilabel)
        assert type(reuters_test.ids) == list
        assert len(reuters_test.ids) == 1000
        assert type(reuters_test.ids[0]) == str
        assert reuters_test.ids[0] == '14826'        

    def test_texts(self, imdb_test, reuters_test):
        # imdb (multiclass)
        assert type(imdb_test.texts) == list
        assert len(imdb_test.texts) == 1000
        assert len(imdb_test.texts[0]) == 897
        # reuters (multilabel)
        assert type(reuters_test.texts) == list
        assert len(reuters_test.texts) == 1000
        assert len(reuters_test.texts[0]) == 4425        

    def test_lemmatized_texts(self, imdb_test, imdb_test_lem):
        assert type(imdb_test_lem.lemmatized_texts) == list
        assert len(imdb_test_lem.lemmatized_texts) == 1000
        assert len(imdb_test_lem.lemmatized_texts[0]) == 807
        with pytest.raises(RuntimeError):
            imdb_test.lemmatized_texts

    def test_X(self, imdb_test, imdb_test_lem):
        # not lemmatized
        assert type(imdb_test.X) == list
        assert len(imdb_test.X) == 1000
        assert len(imdb_test.X[0]) == 897
        # lemmatized
        assert type(imdb_test_lem.X) == list
        assert len(imdb_test_lem.X) == 1000
        assert len(imdb_test_lem.X[0]) == 807

    def test_labels(self, imdb_test, reuters_test):
        # imdb (multiclass)
        assert type(imdb_test.labels) == list
        assert len(imdb_test.labels) == 1000
        assert type(imdb_test.labels[0]) == str
        assert imdb_test.labels[0] == 'negative'
        # reuters (multilabel)
        assert type(reuters_test.labels) == list
        assert len(reuters_test.labels) == 1000
        assert type(reuters_test.labels[0]) == list
        assert type(reuters_test.labels[0][0]) == str
        assert reuters_test.labels[0][0] == 'trade'
        
    def test_y(self, imdb_test, reuters_test):
        # imdb (multiclass)
        assert type(imdb_test.y) == list
        assert len(imdb_test.y) == 1000
        assert type(imdb_test.y[0]) == str
        assert imdb_test.y[0] == 'negative'
        # reuters (multilabel)
        assert type(reuters_test.y) == list
        assert len(reuters_test.y) == 1000
        assert type(reuters_test.y[0]) == list
        assert type(reuters_test.y[0][0]) == str
        assert reuters_test.y[0][0] == 'trade'
        
    def test_is_multilabel(self, imdb_test, reuters_test):
        # imdb (multiclass)
        assert imdb_test.is_multilabel == False
        # reuters (multilabel)
        assert reuters_test.is_multilabel == True
        
    def test_label_columns(self, imdb_test):
        assert imdb_test.label_columns == ['sentiment']

    def test_label_column(self, imdb_test, reuters_test):
        # imdb (multiclass)
        assert imdb_test.label_column == 'sentiment'
        # reuters (multilabel)
        assert reuters_test.label_column == 'topics'

    def test_split(self, imdb_test, reuters_test):
        # imdb (multiclass)
        splits = imdb_test.split(['train', 'test', 'dev'], [0.8, 0.1, 0.1])
        assert type(splits) == Dataset
        assert len(splits) == 3
        assert set(splits.keys()) == set(['train', 'test', 'dev'])
        assert len(splits['train'].ids) == 800
        assert len(splits['test'].ids) == 100
        assert len(splits['dev'].ids) == 100
        assert len(set(splits['train'].ids + splits['test'].ids + splits['dev'].ids)) == 1000
        # reuters (multilabel)
        splits = reuters_test.split(['train', 'test', 'dev'], [0.8, 0.1, 0.1])
        assert type(splits) == Dataset
        assert len(splits) == 3
        assert set(splits.keys()) == set(['train', 'test', 'dev'])
        assert len(splits['train'].ids) == 800
        assert len(splits['test'].ids) == 100
        assert len(splits['dev'].ids) == 100
        assert len(set(splits['train'].ids + splits['test'].ids + splits['dev'].ids)) == 1000 
        
