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

ID_COLUMN = 'id'
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'labels'
LEM_COLUMN = 'text_lem'
LABEL_SEP = '|'

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

    def __init__(self, ids, texts, labels, label_column=None):
        self._data = pd.DataFrame({
            TEXT_COLUMN: texts,
        }, index=pd.Index(ids, name=ID_COLUMN))
        if type(labels) == dict:
            for column, label_values in labels.items():
                self._data[column] = label_values
            self._label_column = label_column
        else:
            self._data[LABEL_COLUMN] = labels
            self._label_column = LABEL_COLUMN

    @staticmethod
    def load(tsv_fpath, label_column=None, label_sep=None):
        column_sep = ',' if os.path.splitext(tsv_fpath)[1] == '.csv' else '\t'
        df = pd.read_csv(tsv_fpath, sep=column_sep, keep_default_na=False, dtype=str)
        if ID_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
            raise KeyError(f"Column '{ID_COLUMN}' or '{TEXT_COLUMN}' not found in {tsv_fpath}")
        if label_column is not None and label_column not in df.columns:
            raise KeyError(f"Column '{label_column}' not found in {tsv_fpath}")
        ids = df[ID_COLUMN]
        texts = df[TEXT_COLUMN]
        label_columns = [ col for col in df.columns if col not in [ID_COLUMN, TEXT_COLUMN, LEM_COLUMN] ]
        labels = df[label_columns]
        label_column_ = label_column
        if label_column_ is None:
            label_column_ = label_columns[0] if len(label_columns) == 1 else LABEL_COLUMN
        if label_column_ not in label_columns:
            raise ValueError("Label column not set and multiple label columns found, but none of them is called 'labels'. Please specify the label_column.")
        if label_sep is not None:
            for col in label_columns:
                pd.options.mode.chained_assignment = None # avoid warning
                labels[col] = labels[col].str.split(label_sep)
                # filter out empty-strings from labels
                labels[col] = labels[col].map(lambda ls: list(filter(lambda e: e != "", ls)))
        return DatasetSplit(ids.tolist(), texts.tolist(), labels.to_dict('list'), label_column)

    @property
    def ids(self):
        return self._data.index.tolist()
    
    @property
    def texts(self):
        return self._data[TEXT_COLUMN].tolist()

    @property
    def lemmatized_texts(self):
        if LEM_COLUMN not in self._data.columns:
            raise RuntimeError("Dataset is not lemmatized")
        return self._data[LEM_COLUMN].tolist()

    @property
    def X(self):
        column = LEM_COLUMN if LEM_COLUMN in self._data.columns else TEXT_COLUMN
        return self._data[column].tolist()

    @property
    def labels(self):
        return self._data[self.label_column].tolist()

    @property
    def y(self):
        return self.labels

    @property
    def is_multilabel(self):
        labels = self.labels
        return False if len(labels) == 0 else type(self.labels[0]) == list

    @property
    def label_columns(self):
        return [ col for col in self._data.columns if col not in [ID_COLUMN, TEXT_COLUMN, LEM_COLUMN] ]
    
    @property
    def label_column(self):
        col = self._label_column
        if col is None:
            if len(self.label_columns) == 1:
                col = self.label_columns[0]
            else:
                col = LABEL_COLUMN
        if col not in self._data.columns:
            raise KeyError(f"Label column '{col}' not found")
        return col

    def to_hf(self):
        columns = [ID_COLUMN, TEXT_COLUMN, self.label_column]
        ds_data = datasets.Dataset.from_pandas(self._data[columns])
        ds_y = datasets.Dataset.from_dict({'y': self._y})
        ds = datasets.concatenate_datasets([ds_data, ds_y], axis=1)
        ds = ds.remove_columns([ID_COLUMN, self.label_column])
        ds = ds.rename_column('y', 'labels')
        ds = ds.cast_column('labels', datasets.Sequence(datasets.Value(dtype='float32', id=None)))
        return ds

    def clean_texts(self):
        self._data[TEXT_COLUMN] = utils.clean_texts(self._data[TEXT_COLUMN])
        
    def lemmatize(self, dic_path):
        lemmatized = utils.lemmatize(self._data[TEXT_COLUMN], dic_path)
        text_idx = self._data.columns.tolist().index(TEXT_COLUMN)
        self._data.insert(text_idx+1, LEM_COLUMN, lemmatized)

    def oversample(self, target_f=np.average):
        _, y, ids = utils.oversample(self.texts, self.y, self.ids, target_f)
        self._data = self._data.loc[ids]
        self._y = y

    def binarize_labels(self):
        lb = MultiLabelBinarizer() if self.is_multilabel else LabelBinarizer()
        y = lb.fit_transform(self.labels)
        if len(lb.classes_) == 2:
            # force one-hot encoding if binary classification, as LabelBinarizer encodes in 1D
            y = np.array([ (1, 0) if l == 0 else (0, 1) for l in y ])
        return y

    def split(self, names, sizes, label_column=None):
        if len(names) != len(sizes):
            raise ValueError("'names' and 'sizes' must contain the same number of items")
        if sum(sizes) > 1.0:
            raise ValueError("Split sizes sum up to a value greater than 1")
        if label_column is None:
            label_column = self.label_column
        if len(self.label_columns) > 1 and label_column is None:
            raise ValueError("Multiple label columns found in the dataset and label_column is undefined, specify the column to use for stratification providing the 'label' argument")

        if self.is_multilabel:
            split_f = iterative_train_test_split
        else:
            split_f = lambda X, y, size: train_test_split(X, y, test_size=size, stratify=np.argmax(y, axis=1))

        y = self.binarize_labels()
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
        ids = data.index.to_numpy()
        texts = data[TEXT_COLUMN].to_numpy()
        labels = data[self.label_column].to_numpy()
        labels = {self.label_column: labels}
        train_split = DatasetSplit(ids, texts, labels)
        return train_split

    def save(self, fpath, sep='|', override=False):
        if not override and os.path.isfile(fpath):
            raise ValueError(f"The given path corresponds to an existing file")
        dir_path = os.path.dirname(fpath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.is_multilabel:
            for col in self.label_columns:
                self._data[col] = self._data[col].str.join(sep)
        self._data.to_csv(fpath, sep='\t', index=True)
