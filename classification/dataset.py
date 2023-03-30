import classification.utils as utils

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import KFold, train_test_split
from skmultilearn.model_selection import iterative_train_test_split

import pandas as pd
import numpy as np
import datasets

import itertools
import logging
import pdb
import os
from collections.abc import MutableMapping

LEM_SUFFIX = '_lem'

class Dataset(MutableMapping):

    def __init__(self, splits, mode='auto'):
        self.splits = splits
        # mode is multilabel?
        if mode not in [None, 'auto', 'multilabel', 'multiclass']:
            raise ValueError("Param 'mode' must be one of: 'auto', 'multilabel', 'multiclass', None")
        if mode == 'multilabel':
            self._is_multilabel = True
        elif mode == 'multiclass':
            self._is_multilabel = False
        elif mode == 'auto' and len(splits) > 0:
            are_multilabel = [ ds.is_multilabel for ds in splits.values() ]
            if len(set(are_multilabel)) != 1:
                logging.warning("Cannot infer mode automatically, values for different splits differ")
                self._is_multilabel = None
            else:
                self._is_multilabel = are_multilabel[0]
        else:
            self._is_multilabel = None
        # binarize labels
        self._binarize_labels()

    def _binarize_labels(self):
        self.label_binarizer = MultiLabelBinarizer() if self._is_multilabel else LabelBinarizer()
        merged_labels = list(itertools.chain(*[ ds.labels for ds in self.splits.values() ]))
        self.label_binarizer.fit(merged_labels)
        for name, ds in self.splits.items():
            pass
            #ds.binarize_labels(self.label_binarizer)

    @staticmethod
    def load(split_files, mode='auto', label_column=None):
        splits = {
            name: DatasetSplit.load(fpath, mode, label_column)
            for name, fpath in split_files.items()
        }
        return Dataset(splits, mode)

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

    def save(self, path, override=False):
        if os.path.isfile(path):
            raise ValueError(f"The given path corresponds to an existing file")
        for name, ds in self.splits.items():
            fpath = os.path.join(path, name+'.tsv')
            ds.save(fpath, override)

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
                if 'text' in col.lower():
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
            # warning
            return self.texts

    @property
    def labels(self):
        return self._data[self._labelc].tolist()

    @property
    def y(self):
        return self.labels

    @property
    def is_multilabel(self):
        return False if len(self.labels) == 0 else type(self.labels[0]) == list

    @property
    def label_column(self):
        return self._labelc

    def to_hf(self):
        ds = datasets.Dataset.from_pandas(self._data)
        ds = ds.add_column('__text__', self.texts)
        ds = ds.remove_columns([ col for col in ds.features.keys() if col not in ['__text__', self._idc, self._labelc] ])
        ds = ds.rename_column('__text__', 'text')
        if self._idc != 'id':
            ds = ds.rename_column(self._idc, 'id')
        if self._labelc != 'labels':
            ds = ds.rename_column(self._labelc, 'labels')
        return ds

    def clean_texts(self):
        for col in self._textc:
            self._data[col] = utils.clean_texts(self._data[col])
        
    def lemmatize(self, dic_path):
        for col in self._textc:
            lemmatized = utils.lemmatize(self._data[col], dic_path)
            text_idx = self._data.columns.tolist().index(col)
            self._data.insert(text_idx+1, col+LEM_SUFFIX, lemmatized)

    def oversample(self, target_f=np.average):
        _, _, ids = utils.oversample(self.texts, self._binarize_labels(), self.ids, target_f)
        self._data = self._data.loc[ids]

    def split(self, names, sizes):
        if len(names) != len(sizes):
            raise ValueError("'names' and 'sizes' must contain the same number of items")
        if sum(sizes) > 1.0:
            raise ValueError("Split sizes sum up to a value greater than 1")

        if self.is_multilabel:
            split_f = iterative_train_test_split
        else:
            split_f = lambda X, y, size: train_test_split(X, y, test_size=size, stratify=np.argmax(y, axis=1))

        y = self._binarize_labels()
        rem_inds = np.array(range(len(self.y)))

        specs = [ (name, size) for name, size in zip(names, sizes) if size > 0.0 ]
        splits = {}
        for i, (split_name, split_size) in enumerate(specs):
            if i == len(specs)-1:
                split_inds = rem_inds
            else:
                size = split_size / (len(rem_inds) / len(self._data.index))
                res = split_f(rem_inds.reshape((-1,1)), y[rem_inds,:], 1.0-size)
                split_inds = res[0].squeeze()
                rem_inds = np.array([ idx for idx in rem_inds if idx not in split_inds ])
            splits[split_name] = self._build_from_indices(split_inds)

        return Dataset(splits)

    def kfold(self, k):
        kfold = KFold(n_splits=k, shuffle=True)
        for i, (train_index, test_index) in enumerate(kfold.split(self.y)):
            train_split = self._build_from_indices(train_index)
            test_split = self._build_from_indices(test_index)
            yield i, (train_split, test_split)

    def _build_from_indices(self, indices):
        data = self._data.iloc[indices]
        return DatasetSplit(data.reset_index(), id_column=self._idc, text_columns=self._textc,
                            label_column=self._labelc)

    def _binarize_labels(self):
        lb = MultiLabelBinarizer() if self.is_multilabel else LabelBinarizer()
        y = lb.fit_transform(self.labels)
        if len(lb.classes_) == 2:
            # force one-hot encoding if binary classification, as LabelBinarizer encodes in 1D
            y = np.array([ (1, 0) if l == 0 else (0, 1) for l in y ])
        return y

    def save(self, fpath, label_sep='|', override=False):
        if not override and os.path.isfile(fpath):
            raise ValueError(f"The given path corresponds to an existing file")
        dir_path = os.path.dirname(fpath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.is_multilabel:
            self._data[self._labelc] = self._data[self._labelc].str.join(label_sep)
        column_sep = ',' if os.path.splitext(fpath)[1] == '.csv' else '\t'
        self._data.to_csv(fpath, sep=column_sep, index=True)
