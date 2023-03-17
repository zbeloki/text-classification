import general_functions as helper

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

import pandas as pd
import numpy as np

import itertools
import pdb

ID_COLUMN = 'id'
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'labels'
LEM_COLUMN = 'text_lem'
LABEL_SEP = '|'

class Dataset:

    def __init__(self, splits, multilabel):
        self.splits = splits
        self.multilabel = multilabel
        self._binarize_labels()

    def _binarize_labels(self):
        self.label_binarizer = MultiLabelBinarizer() if self.multilabel else LabelBinarizer()
        merged_labels = list(itertools.chain(*[ ds.labels for ds in self.splits.values() ]))
        self.label_binarizer.fit(merged_labels)
        for name, ds in self.splits.items():
            ds.binarize_labels(self.label_binarizer)

    @staticmethod
    def load(split_files, multilabel, label_column=LABEL_COLUMN):
        splits = {
            name: DatasetSplit.load(fpath, multilabel, label_column)
            for name, fpath in split_files.items()
        }
        return Dataset(splits, multilabel)

    def lemmatize(self, dic_path):
        for name, ds in self.splits.items():
            ds.lemmatize(dic_path)

    def to_hf(self):
        pass

    def __getitem__(self, split_name):
        return self.splits[split_name]

    def __contains__(self, split_name):
        return split_name in self.splits

class DatasetSplit:

    def __init__(self, ids, texts, labels):
        self._data = pd.DataFrame({
            TEXT_COLUMN: texts,
            LABEL_COLUMN: labels,
        }, index=ids)
        self._y = None

    def binarize_labels(self, label_binarizer):
        self._y = label_binarizer.transform(self._data[LABEL_COLUMN])

    @staticmethod
    def load(tsv_fpath, multilabel, label_column=LABEL_COLUMN):
        df = pd.read_csv(tsv_fpath, sep='\t', keep_default_na=False, dtype=str)
        if ID_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
            raise KeyError(f"Column '{ID_COLUMN}' or '{TEXT_COLUMN}' not found in {tsv_fpath}")
        if label_column not in df.columns:
            raise KeyError(f"Column '{label_column}' not found in {tsv_fpath}")
        ids = df[ID_COLUMN]
        texts = df[TEXT_COLUMN]
        labels = df[label_column]
        if multilabel:
            labels = labels.str.split(LABEL_SEP)
            # filter out empty-strings from labels
            labels = labels.map(lambda ls: list(filter(lambda e: e != "", ls)))
        return DatasetSplit(ids.to_numpy(), texts.to_numpy(), labels.to_numpy())

    @property
    def ids(self):
        return self._data.index.to_numpy()
    
    @property
    def X(self):
        column = LEM_COLUMN if LEM_COLUMN in self._data.columns else TEXT_COLUMN
        return self._data[column].to_numpy()

    @property
    def texts(self):
        return self._data[TEXT_COLUMN].to_numpy()

    @property
    def lemmatized_texts(self):
        if LEM_COLUMN not in self._data.columns:
            raise RuntimeError("Dataset is not lemmatized")
        return self._data[TEXT_COLUMN].to_numpy()

    @property
    def y(self):
        if self._y is None:
            raise RuntimeError("Dataset labels are not binarized")
        return self._y

    @property
    def labels(self):
        return self._data[LABEL_COLUMN].to_numpy().tolist()

    def lemmatize(self, dic_path):
        self._data[LEM_COLUMN] = helper.lemmatize(self._data[TEXT_COLUMN], dic_path)

    def oversample(self, target_f=np.average):
        _, y, ids = helper.oversample(self.texts, self.y, self.ids, target_f)
        self._data = self._data.loc[ids]
        self._y = y
