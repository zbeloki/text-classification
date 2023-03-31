from classification.dataset import Dataset, DatasetSplit

import pandas as pd
import numpy as np
import datasets

import pytest
import pathlib
import tempfile
import os
import pdb

@pytest.fixture(scope='session')
def datadir():
    return pathlib.Path(__file__).parent.resolve() / 'data'

@pytest.fixture(scope='class')
def imdb_test(datadir):
    return DatasetSplit.load(datadir/"imdb.test.tsv", label_column='sentiment')

@pytest.fixture(scope='class')
def imdb_test_lem(datadir):
    ds = DatasetSplit.load(datadir/"imdb.test.tsv", label_column='sentiment')
    hs_dic = os.path.join(datadir, 'en_US')
    ds.lemmatize(hs_dic)
    return ds

@pytest.fixture(scope='class')
def reuters_train(datadir):
    return DatasetSplit.load(datadir/"reuters.train.csv",
                             id_column='article_id',
                             text_columns=['title', 'body'],
                             label_column='topics',
                             label_sep='|')

@pytest.fixture(scope='class')
def reuters_test(datadir):
    return DatasetSplit.load(datadir/"reuters.test.csv",
                             id_column='article_id',
                             text_columns=['title', 'body'],
                             label_column='topics',
                             label_sep='|')

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

class TestDatasetSplit:

    def test_decide_columns(self):
        df = pd.DataFrame({'IDS': [], 'texts': [], 'text_body': [], 'topic_label': [], 'other': []})
        idc, textc, labelc = DatasetSplit._decide_columns(df.columns, None, None, None)
        assert (idc, textc, labelc) == ('IDS', ['texts', 'text_body'], 'topic_label')
        df = pd.DataFrame({'codes': [], 'title': [], 'body': [], 'sentiment': []})
        idc, textc, labelc = DatasetSplit._decide_columns(df.columns, 'codes', ['body'], 'sentiment')
        assert (idc, textc, labelc) == ('codes', ['body'], 'sentiment')
        with pytest.raises(RuntimeError):
            df = pd.DataFrame({'codes': [], 'text': [], 'labels': []})
            DatasetSplit._decide_columns(df.columns, None, None, None)
        with pytest.raises(RuntimeError):
            df = pd.DataFrame({'ids': [], 'test': [], 'labels': []})
            DatasetSplit._decide_columns(df.columns, 'ids', None, 'labels')
        with pytest.raises(ValueError):
            df = pd.DataFrame({'ids': [], 'test': [], 'labels': []})
            DatasetSplit._decide_columns(df.columns, 'id', ['test'], 'labels')
           
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
        assert len(reuters_test.texts[0]) == 4474        

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
        
    def test_label_column(self, imdb_test, reuters_test):
        # imdb (multiclass)
        assert imdb_test.label_column == 'sentiment'
        # reuters (multilabel)
        assert reuters_test.label_column == 'topics'

    def test_to_hf(self, imdb_test, reuters_test):
        # imdb (multiclass)
        ds = imdb_test.to_hf()
        assert ds.features.keys() == set(['id', 'text', 'labels'])
        assert ds.features['id'] == datasets.Value(dtype='string')
        assert ds.features['text'] == datasets.Value(dtype='string')
        assert ds.features['labels'] == datasets.Value(dtype='string')
        assert len(ds) == 1000
        assert ds['id'][0] == 'r30773'
        assert len(ds['text'][0]) == 897
        assert ds['labels'][0] == 'negative'
        # reuters (multilabel)
        ds = reuters_test.to_hf()
        assert ds.features.keys() == set(['id', 'text', 'labels'])
        assert ds.features['id'] == datasets.Value(dtype='string')
        assert ds.features['text'] == datasets.Value(dtype='string')
        assert ds.features['labels'] == datasets.Sequence(feature=datasets.Value(dtype='string'))
        assert len(ds) == 1000
        assert ds['id'][0] == '14826'
        assert len(ds['text'][0]) == 4474
        assert ds['labels'][0] == ['trade']

    def test_clean_texts(self):
        body_html = "The U.S. has said it will impose <span><i>300 mln</i> dlrs </span>of tariffs on imports of Japanese electronics goods on <span>April 17</span>, in retaliation for Japan's alleged failure to stick to a pact not to sell semiconductors<br/><br/>on world markets at below cost."
        body_clean = "The U.S. has said it will impose 300 mln dlrs of tariffs on imports of Japanese electronics goods on April 17 , in retaliation for Japan's alleged failure to stick to a pact not to sell semiconductors on world markets at below cost."
        ds = DatasetSplit(pd.DataFrame({
            'id': ['1'],
            'text_title': ["<h1>ASIAN EXPORTERS FEAR DAMAGE FROM U.S.</h1>"],
            'text_body': [body_html],
            'label': ['0'],
        }))
        ds.clean_texts()
        assert ds._data['text_title'][0] == "ASIAN EXPORTERS FEAR DAMAGE FROM U.S."
        assert ds._data['text_body'][0] == body_clean

    def test_oversample(self, small_mc, small_ml):
        # multiclass
        small_mc.oversample()
        assert abs(len(small_mc.labels) - 12) <= 1
        assert small_mc.labels.count('A') == 3 # 1x3
        assert small_mc.labels.count('B') == 4 # 2x2
        assert small_mc.labels.count('C') == 6 # 6x1
        small_mc.oversample(np.max)
        assert abs(len(small_mc.labels) - 18) <= 2
        assert small_mc.labels.count('A') == 6 # 3x2
        assert small_mc.labels.count('B') == 8 # 4x2
        assert small_mc.labels.count('C') == 6 # 6x1
        # multilabel
        small_ml.oversample()
        assert abs(len(small_ml.labels) - 9) <= 0
        assert len([ 1 for labels in small_ml.labels if 'A' in labels ]) == 5
        assert len([ 1 for labels in small_ml.labels if 'B' in labels ]) == 6
        assert len([ 1 for labels in small_ml.labels if 'C' in labels ]) == 4
        
    def test_split(self, imdb_test, reuters_test):
        # imdb (multiclass)
        splits = imdb_test.split(['train', 'test', 'dev'], [0.8, 0.1, 0.1])
        assert type(splits) == Dataset
        assert len(splits) == 3
        assert splits.keys() == set(['train', 'test', 'dev'])
        assert len(splits['train'].ids) == 800
        assert len(splits['test'].ids) == 100
        assert len(splits['dev'].ids) == 100
        assert len(set(splits['train'].ids + splits['test'].ids + splits['dev'].ids)) == 1000
        assert splits['train'].ids[0].startswith('r')
        # reuters (multilabel)
        splits = reuters_test.split(['train', 'test', 'dev'], [0.8, 0.1, 0.1])
        assert type(splits) == Dataset
        assert len(splits) == 3
        assert splits.keys() == set(['train', 'test', 'dev'])
        assert len(splits['train'].ids) == 800
        assert len(splits['test'].ids) == 100
        assert len(splits['dev'].ids) == 100
        assert len(set(splits['train'].ids + splits['test'].ids + splits['dev'].ids)) == 1000

    def test_kfold(self, imdb_test):
        folds = list(imdb_test.kfold(3))
        train_ids = lambda fold: fold[1][0].ids
        test_ids = lambda fold: fold[1][1].ids
        assert len(set(test_ids(folds[0]) + test_ids(folds[1]) + test_ids(folds[2]))) == 1000
        for fold in folds:
            assert len(set(train_ids(fold) + test_ids(fold))) == 1000
            assert abs(len(train_ids(fold)) - 666) <= 1
            assert abs(len(test_ids(fold)) - 334) <= 1

    def test_save(self, imdb_test, reuters_test):
        # imdb (multiclass)
        with tempfile.TemporaryDirectory() as path:
            fpath = os.path.join(path, "test.csv")
            imdb_test.save(fpath)
            assert os.path.exists(fpath)
            df = pd.read_csv(fpath, keep_default_na=False)
            assert set(df.columns) == set(['id', 'text', 'sentiment'])
            assert len(df) == len(imdb_test.ids)
        # reuters (multilabel)
        with tempfile.TemporaryDirectory() as path:
            fpath = os.path.join(path, "test.tsv")
            reuters_test.save(fpath, label_sep=',')
            assert os.path.exists(fpath)
            df = pd.read_csv(fpath, sep='\t', keep_default_na=False)
            assert set(df.columns) == set(['article_id', 'document_date', 'people', 'orgs', 'places', 'exchanges', 'topics', 'title', 'body'])
            assert len(df) == len(reuters_test.ids)
            assert set(df.iloc[2].topics.split(',')) == set(['crude', 'nat-gas'])
