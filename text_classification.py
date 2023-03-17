import general_functions as helper

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.metrics import fbeta_score, precision_score, recall_score

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

class TextClassifier:

    def __init__(self, model):
        self._model = model

    @staticmethod
    def load(model_path):
        model_fpath = os.path.join(model_path, 'model.joblib')
        return joblib.load(model_fpath)

    def classify(self, texts, threshold=0.5, top_n=None):
        probas = self._model.predict_proba(texts)
        y = self.predict(texts, threshold=threshold, top_n=top_n)
        result = []
        for i in range(len(y)):
            row_probas = probas[i][np.argwhere(y[i] == 1).flatten()].tolist()
            row_labels = self._model.classes_[np.argwhere(y[i] == 1).flatten()].tolist()
            row_result = sorted(zip(row_labels, row_probas), key=lambda e:e[1], reverse=True)
            result.append(row_result)
            
        return result

    def predict(self, texts, threshold=0.5, top_n=None):
        probas = self._model.predict_proba(texts)
        probas[probas < threshold] = 0
        if top_n is not None:
            threshold_probas = -np.sort(-probas)[:, top_n]
            # add one dimension
            threshold_probas = threshold_probas[..., np.newaxis]
            probas[probas <= threshold_probas] = 0
        probas[probas > 0] = 1
        
        return probas

def evaluate(estimator, X, y, threshold=None, top_n=None, beta=1):

    kwargs = {}
    if threshold is not None:
        kwargs['threshold'] = threshold
    if top_n is not None:
        kwargs['top_n'] = top_n
    text_classifier = TextClassifier(estimator)
    y_pred = text_classifier.predict(X, **kwargs)
    
    f = fbeta_score(y, y_pred, beta=beta, average='micro')
    p = precision_score(y, y_pred, average='micro')
    r = recall_score(y, y_pred, average='micro')

    return f, p, r
