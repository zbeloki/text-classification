from classification.dataset import Dataset, DatasetSplit

import pandas as pd
import numpy as np
import datasets

import pytest
import pathlib
import tempfile
import os
import pdb

class TestDataset:

    def test_load(self, imdb):
        assert imdb.keys() == set(['train', 'test', 'dev'])
        assert len(imdb['test'].ids) == 20

    def test_clean_texts(self, imdb):
        assert '<br />' in imdb['test'].texts[3]
        imdb.clean_texts()
        assert '<br />' not in imdb['test'].texts[3]

    def test_lemmatize(self, imdb_func, hunspell):
        assert len(imdb_func['test'].X[0]) == 897
        imdb_func.lemmatize(hunspell)
        assert len(imdb_func['test'].X[0]) == 807

    def test_save(self, imdb):
        with tempfile.TemporaryDirectory() as path:
            imdb.save(path, ext='csv')
            test_fpath = os.path.join(path, 'test.csv')
            train_fpath = os.path.join(path, 'train.csv')
            assert os.path.exists(test_fpath)
            assert os.path.exists(train_fpath)
            df = pd.read_csv(test_fpath, keep_default_na=False)
            assert set(df.columns) == set(['id', 'text', 'sentiment'])
            assert len(df) == len(imdb['test'].ids)        

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
           
    def test_ids(self, imdb_test, toxic_test):
        # imdb (binary multiclass)
        assert type(imdb_test.ids) == list
        assert len(imdb_test.ids) == 20
        assert type(imdb_test.ids[0]) == str
        assert imdb_test.ids[0] == 'r30773'
        # toxic (multilabel)
        assert type(toxic_test.ids) == list
        assert len(toxic_test.ids) == 20
        assert type(toxic_test.ids[0]) == str
        assert toxic_test.ids[0] == '6728190ab6bb7bb0'        

    def test_texts(self, imdb_test, toxic_test):
        # imdb (binary multiclass)
        assert type(imdb_test.texts) == list
        assert len(imdb_test.texts) == 20
        assert len(imdb_test.texts[0]) == 897
        # toxic (multilabel)
        assert type(toxic_test.texts) == list
        assert len(toxic_test.texts) == 20
        assert len(toxic_test.texts[0]) == 1485        

    def test_lemmatized_texts(self, imdb_test, imdb_lem):
        imdb_test_lem = imdb_lem['test']
        assert type(imdb_test_lem.lemmatized_texts) == list
        assert len(imdb_test_lem.lemmatized_texts) == 20
        assert len(imdb_test_lem.lemmatized_texts[0]) == 807
        with pytest.raises(RuntimeError):
            imdb_test.lemmatized_texts

    def test_X(self, imdb_test, imdb_lem):
        imdb_test_lem = imdb_lem['test']
        # not lemmatized
        assert type(imdb_test.X) == list
        assert len(imdb_test.X) == 20
        assert len(imdb_test.X[0]) == 897
        # lemmatized
        assert type(imdb_test_lem.X) == list
        assert len(imdb_test_lem.X) == 20
        assert len(imdb_test_lem.X[0]) == 807

    def test_labels(self, imdb_test, toxic_test):
        # imdb (binary multiclass)
        assert type(imdb_test.labels) == list
        assert len(imdb_test.labels) == 20
        assert type(imdb_test.labels[0]) == str
        assert imdb_test.labels[0] == 'negative'
        # toxic (multilabel)
        assert type(toxic_test.labels) == list
        assert len(toxic_test.labels) == 20
        assert type(toxic_test.labels[0]) == list
        assert type(toxic_test.labels[4][0]) == str
        assert set(toxic_test.labels[4]) == set(['toxic', 'obscene', 'insult'])
        
    def test_y(self, imdb_test, toxic_test):
        # imdb (binary multiclass)
        assert type(imdb_test.y()) == np.ndarray
        assert len(imdb_test.y()) == 20
        assert type(imdb_test.y()[0]) == np.ndarray
        assert np.allclose(imdb_test.y()[0], [1, 0])
        # toxic (multilabel)
        assert type(toxic_test.y()) == np.ndarray
        assert len(toxic_test.y()) == 20
        assert type(toxic_test.y()[0]) == np.ndarray
        assert type(toxic_test.y()[4][0]) == np.int64
        assert np.allclose(toxic_test.y()[4], [0, 1, 1, 0, 1])
        
    def test_is_multilabel(self, imdb_test, toxic_test):
        # imdb (binary multiclass)
        assert imdb_test.is_multilabel == False
        # toxic (multilabel)
        assert toxic_test.is_multilabel == True
        
    def test_label_column(self, imdb_test, toxic_test):
        # imdb (binary multiclass)
        assert imdb_test.label_column == 'sentiment'
        # toxic (multilabel)
        assert toxic_test.label_column == 'tags'

    def test_to_hf(self, imdb_test, toxic_test):
        # imdb (binary multiclass)
        ds = imdb_test.to_hf()
        assert ds.features.keys() == set(['id', 'text', 'labels'])
        assert ds.features['id'] == datasets.Value(dtype='string')
        assert ds.features['text'] == datasets.Value(dtype='string')
        assert ds.features['labels'] == datasets.Value(dtype='string')
        assert len(ds) == 20
        assert ds['id'][0] == 'r30773'
        assert len(ds['text'][0]) == 897
        assert ds['labels'][0] == 'negative'
        # toxic (multilabel)
        ds = toxic_test.to_hf()
        assert ds.features.keys() == set(['id', 'text', 'labels'])
        assert ds.features['id'] == datasets.Value(dtype='string')
        assert ds.features['text'] == datasets.Value(dtype='string')
        assert ds.features['labels'] == datasets.Sequence(feature=datasets.Value(dtype='string'))
        assert len(ds) == 20
        assert ds['id'][0] == '6728190ab6bb7bb0'
        assert len(ds['text'][0]) == 1485
        assert ds['labels'][4] == ['toxic', 'obscene', 'insult']

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
        
    def test_split(self, imdb_test, toxic_test):
        # imdb (binary multiclass)
        splits = imdb_test.split(['train', 'test', 'dev'], [0.8, 0.1, 0.1])
        assert type(splits) == Dataset
        assert len(splits) == 3
        assert splits.keys() == set(['train', 'test', 'dev'])
        assert len(splits['train'].ids) == 16
        assert len(splits['test'].ids) == 2
        assert len(splits['dev'].ids) == 2
        assert len(set(splits['train'].ids + splits['test'].ids + splits['dev'].ids)) == 20
        assert splits['train'].ids[0].startswith('r')
        # toxic (multilabel)
        splits = toxic_test.split(['train', 'test', 'dev'], [0.8, 0.1, 0.1])
        assert type(splits) == Dataset
        assert len(splits) == 3
        assert splits.keys() == set(['train', 'test', 'dev'])
        assert len(splits['train'].ids) == 16
        assert len(splits['test'].ids) == 2
        assert len(splits['dev'].ids) == 2
        assert len(set(splits['train'].ids + splits['test'].ids + splits['dev'].ids)) == 20

    def test_kfold(self, imdb_test):
        folds = list(imdb_test.kfold(3))
        train_ids = lambda fold: fold[1][0].ids
        test_ids = lambda fold: fold[1][1].ids
        assert len(set(test_ids(folds[0]) + test_ids(folds[1]) + test_ids(folds[2]))) == 20
        for fold in folds:
            assert len(set(train_ids(fold) + test_ids(fold))) == 20
            assert abs(len(train_ids(fold)) - 13) <= 1
            assert abs(len(test_ids(fold)) - 7) <= 1

    def test_save(self, imdb_test, toxic_test):
        # imdb (binary multiclass)
        with tempfile.TemporaryDirectory() as path:
            fpath = os.path.join(path, "test.csv")
            imdb_test.save(fpath)
            assert os.path.exists(fpath)
            df = pd.read_csv(fpath, keep_default_na=False)
            assert set(df.columns) == set(['id', 'text', 'sentiment'])
            assert len(df) == len(imdb_test.ids)
        # toxic (multilabel)
        with tempfile.TemporaryDirectory() as path:
            fpath = os.path.join(path, "test.tsv")
            toxic_test.save(fpath, label_sep=',')
            assert os.path.exists(fpath)
            df = pd.read_csv(fpath, sep='\t', keep_default_na=False)
            assert set(df.columns) == set(['id', 'comment_text', 'tags'])
            assert len(df) == len(toxic_test.ids)
            assert set(df.iloc[4].tags.split(',')) == set(['toxic', 'obscene', 'insult'])
