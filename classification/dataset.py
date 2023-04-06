import classification.utils as utils

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import KFold, train_test_split
from skmultilearn.model_selection import iterative_train_test_split

import pandas as pd
import numpy as np
import datasets

from collections.abc import MutableMapping
import logging
import pdb
import os

LEM_SUFFIX = '_lem'

class Dataset(MutableMapping):

    def __init__(self, splits, mode='auto'):
        self.splits = splits

    @staticmethod
    def load(split_files, id_column=None, text_columns=None, label_column=None, label_sep=None):
        splits = {
            name: DatasetSplit.load(fpath, id_column, text_columns, label_column, label_sep)
            for name, fpath in split_files.items()
        }
        return Dataset(splits)

    def clean_texts(self):
        for name, ds in self.splits.items():
            ds.clean_texts()      

    def lemmatize(self, dic_path):
        for name, ds in self.splits.items():
            ds.lemmatize(dic_path)

    def __getitem__(self, split_name):
        return self.splits[split_name]

    def __setitem__(self, split_name, split):
        self.splits[split_name] = split

    def __delitem__(self, key):
        del self.splits[key]

    def __contains__(self, split_name):
        return split_name in self.splits

    def __iter__(self):
        return iter(self.splits)

    def __len__(self):
        return len(self.splits)

    def save(self, path, label_sep='|', ext='tsv', override=False):
        if os.path.isfile(path):
            raise ValueError(f"The given path corresponds to an existing file")
        ext = ext.strip('.')
        if ext not in ['tsv', 'csv']:
            raise ValueError(f"ext must be one of 'tsv' or 'csv'")
        for name, ds in self.splits.items():
            fpath = os.path.join(path, name+'.'+ext)
            ds.save(fpath, label_sep, override)

class DatasetSplit:

    def __init__(self, df, id_column=None, text_columns=None, label_column=None):

        idc, textc, labelc = self._decide_columns(df.columns, id_column, text_columns, label_column)
        self._data = df
        self._data.set_index(idc, inplace=True)
        self._idc = idc
        self._textc = textc
        self._labelc = labelc
        
    @classmethod
    def load(cls, fpath, id_column=None, text_columns=None, label_column=None, label_sep=None):
        
        column_sep = ',' if os.path.splitext(fpath)[1] == '.csv' else '\t'
        df = pd.read_csv(fpath, sep=column_sep, keep_default_na=False, dtype=str)
        idc, textc, labelc = cls._decide_columns(df.columns, id_column, text_columns, label_column)

        if label_sep is not None:
            pd.options.mode.chained_assignment = None # avoid warning
            df[labelc] = df[labelc].str.split(label_sep)
            # filter out empty-strings from labels
            df[labelc] = df[labelc].map(lambda ls: list(filter(lambda e: e != "", ls)))
            
        return DatasetSplit(df, idc, textc, labelc)

    @staticmethod
    def _decide_columns(columns, id_column, text_columns, label_column):
        
        if id_column is None:
            for col in columns:
                if col.lower() in ['id', 'ids']:
                    id_column = col
                    break
            if id_column is None:
                raise RuntimeError("Could not infer the column for IDs. Provide 'id_column'.")
        elif id_column not in columns:
            raise ValueError(f"Column '{id_column}' provided in id_column doesn't exist")
            
        if text_columns is None:
            text_columns = []
            for col in columns:
                if 'text' in col.lower() and '_lem' not in col:
                    text_columns.append(col)
            if len(text_columns) == 0:
                raise RuntimeError("Could not infer the column(s) for texts. Provide 'text_columns'.")
        else:
            for col in text_columns:
                if col not in columns:
                    raise ValueError(f"Column '{col}' provided in text_columns doesn't exist")
            
        if label_column is None:
            for col in columns:
                if 'label' in col.lower():
                    label_column = col
                    break
            if label_column is None:
                raise RuntimeError("Could not infer the column for labels. Provide 'label_column'.")
        elif label_column not in columns:
            raise ValueError(f"Column '{label_column}' provided in label_column doesn't exist")
            
        return id_column, text_columns, label_column

    @property
    def ids(self):
        return self._data.index.tolist()
    
    @property
    def texts(self):
        texts = []
        concat_texts = lambda row: ' '.join([ row[textc] for textc in self._textc ])
        texts = self._data.apply(concat_texts, axis=1)
        return texts.tolist()

    @property
    def lemmatized_texts(self):
        try:
            texts = []
            concat_texts = lambda row: ' '.join([ row[textc+LEM_SUFFIX] for textc in self._textc ])
            texts = self._data.apply(concat_texts, axis=1)
            return texts.tolist()
        except KeyError:
            raise RuntimeError(f"At least one column from {', '.join(self._textc)} is not lemmatized")

    @property
    def X(self):
        try:
            return self.lemmatized_texts
        except RuntimeError:
            return self.texts

    @property
    def labels(self):
        return self._data[self._labelc].tolist()

    @property
    def is_multilabel(self):
        return False if len(self.labels) == 0 else type(self.labels[0]) == list

    @property
    def label_column(self):
        return self._labelc

    @property
    def classes(self):
        classes = set()
        for lbs in self.labels:
            if not self.is_multilabel:
                lbs = [lbs]
            classes.update(lbs)
        return sorted(classes)

    @property
    def n_classes(self):
        classes = set()
        for labels in self.labels:
            if self.is_multilabel:
                classes.update(labels)
            else:
                classes.add(labels)
        return len(classes)

    def y(self, label_binarizer=None):
        # OHV if multilabel else class indices
        ohv = self.ohv(label_binarizer)
        if self.is_multilabel:
            return ohv
        else:
            return np.argmax(ohv, axis=1)

    def ohv(self, label_binarizer=None):
        # One-Hot Vector
        if label_binarizer is None:
            label_binarizer = self.create_label_binarizer()
        ohv = label_binarizer.transform(self.labels)
        if len(label_binarizer.classes_) == 2:
            ohv = np.array([ (1, 0) if l == 0 else (0, 1) for l in ohv ])
        return ohv

    def to_hf(self, label_binarizer=None):
        ds = datasets.Dataset.from_pandas(self._data)
        ds = ds.add_column('__text__', self.texts)
        ds = ds.remove_columns([ col for col in ds.features.keys() if col not in ['__text__', self._idc, self._labelc] ])
        ds = ds.rename_column('__text__', 'text')
        if self._idc != 'id':
            ds = ds.rename_column(self._idc, 'id')
        if self._labelc != 'labels':
            ds = ds.rename_column(self._labelc, 'labels')
        # prepare labels
        class_label = datasets.ClassLabel(names=self.classes)
        if self.is_multilabel:
            if label_binarizer is None:
                # log warning!
                label_binarizer = self.create_label_binarizer()
            def one_hot_encode(batch):
                batch['labels'] = label_binarizer.transform(batch['labels'])
                return batch
            ds = ds.map(one_hot_encode, batched=True)
            class_label = datasets.Sequence(datasets.Value(dtype='float32'))
        ds = ds.cast_column('labels', class_label)
        return ds

    def create_label_binarizer(self):
        label_binarizer = MultiLabelBinarizer() if self.is_multilabel else LabelBinarizer()
        label_binarizer.fit(self.labels)
        return label_binarizer

    def clean_texts(self):
        for col in self._textc:
            self._data[col] = utils.clean_texts(self._data[col])
        
    def lemmatize(self, dic_path):
        for col in self._textc:
            lemmatized = utils.lemmatize(self._data[col], dic_path)
            text_idx = self._data.columns.tolist().index(col)
            self._data.insert(text_idx+1, col+LEM_SUFFIX, lemmatized)

    def oversample(self, target_f=np.average):
        _, _, ids = utils.oversample(self.texts, self.ohv(), target_f)
        self._data = self._data.iloc[ids]

    def split(self, names, sizes):
        if len(names) != len(sizes):
            raise ValueError("'names' and 'sizes' must contain the same number of items")
        if sum(sizes) > 1.0:
            raise ValueError("Split sizes sum up to a value greater than 1")

        if self.is_multilabel:
            split_f = iterative_train_test_split
        else:
            split_f = lambda X, y, size: train_test_split(X, y, test_size=size, stratify=np.argmax(y, axis=1))

        rem_inds = np.array(range(len(self.y())))

        specs = [ (name, size) for name, size in zip(names, sizes) if size > 0.0 ]
        splits = {}
        for i, (split_name, split_size) in enumerate(specs):
            if i == len(specs)-1:
                split_inds = rem_inds
            else:
                size = split_size / (len(rem_inds) / len(self._data.index))
                res = split_f(rem_inds.reshape((-1,1)), self.ohv()[rem_inds,:], 1.0-size)
                split_inds = res[0].squeeze()
                rem_inds = np.array([ idx for idx in rem_inds if idx not in split_inds ])
            splits[split_name] = self._build_from_indices(split_inds)

        return Dataset(splits)

    def kfold(self, k):
        kfold = KFold(n_splits=k, shuffle=True)
        for i, (train_index, test_index) in enumerate(kfold.split(self.y())):
            train_split = self._build_from_indices(train_index)
            test_split = self._build_from_indices(test_index)
            yield i, (train_split, test_split)

    def _build_from_indices(self, indices):
        data = self._data.iloc[indices]
        return DatasetSplit(data.reset_index(), id_column=self._idc, text_columns=self._textc,
                            label_column=self._labelc)

    def save(self, fpath, label_sep='|', override=False):
        if not override and os.path.isfile(fpath):
            raise ValueError(f"The given path corresponds to an existing file")
        dir_path = os.path.dirname(fpath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        data = self._data.copy()
        if self.is_multilabel:
            data[self._labelc] = data[self._labelc].str.join(label_sep)
        column_sep = ',' if os.path.splitext(fpath)[1] == '.csv' else '\t'
        data.to_csv(fpath, sep=column_sep, index=True)
