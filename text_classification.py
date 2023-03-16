import general_functions as helper

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, precision_score, recall_score

import pandas as pd
import numpy as np
import hunspell  # libhunspell-dev
from nltk.tokenize import RegexpTokenizer
import tqdm

import itertools
import pdb

ID_COLUMN = 'id'
TEXT_COLUMN = 'text'
LABEL_SEP = '|'

class Dataset:

    def __init__(self, multilabel=False):
        self.splits = {}
        self.multilabel = multilabel
        self.label_binarizer = MultiLabelBinarizer() if multilabel else LabelBinarizer()

    def binarize_labels(self):
        merged_labels = itertools.chain(*[ ds.labels for ds in self.splits.values() ])
        self.label_binarizer.fit(merged_labels)
        for name, ds in self.splits.items():
            ds.y = self.label_binarizer.transform(ds.labels)
            ds.labels = None

    def lemmatize(self, dic_path):
        for name, ds in self.splits.items():
            ds.lemmatize(dic_path)

class DatasetSplit:

    def __init__(self):
        self.ids = None
        self.X = None
        self.y = None
        self.X_lem = None
        self.labels = None

    def lemmatize(self, dic_path):
        self.X_lem = helper.lemmatize(self.X)

class TextClassifier:

    def __init__(self, model, mlb):
        self._model = model
        self._mlb = mlb

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

def load_dataset_split(tsv_fpath, label_column):

    df = pd.read_csv(tsv_fpath, sep='\t', keep_default_na=False, dtype=str)

    if ID_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
        raise KeyError(f"Column '{ID_COLUMN}' or '{TEXT_COLUMN}' not found in {tsv_fpath}")
    if len(df.columns) < 3:
        raise ValueError(f"TSV file must contain at least one column with labels")

    df = df.set_index(ID_COLUMN)

    label_columns = [ c for c in df.columns if c != TEXT_COLUMN ]
    if label_column is None:
        label_column = label_columns[0]

    labels = df[label_column].str.split(LABEL_SEP)

    ds = DatasetSplit()
    ds.ids = df.index.to_numpy()
    ds.X = df.text.to_numpy()
    ds.labels = labels.tolist()
    pdb.set_trace()
    
    return ds

def binarize_labels(dataset, multilabel=False):

    merged_labels = itertools.chain(*[ ds.labels for ds in dataset.values() ])
    single_example_labels = list(dataset.values())[0].labels[0]

    if multilabel:
        if type(single_example_labels) != list:
            raise TypeError("Labels are not multilabel")
        label_binarizer = MultiLabelBinarizer()
    else:
        if type(single_example_labels) == list:
            raise TypeError("Labels are multilabel")
        label_binarizer = LabelBinarizer()
        
    label_binarizer.fit(merged_labels)

    for name, ds in dataset.items():
        ds.y = label_binarizer.transform(ds.labels)
        ds.label_binarizer = label_binarizer
        ds.labels = None

    return label_binarizer

def lemmatize(ds, dic_path):

  hs = hunspell.HunSpell(dic_path+'.dic', dic_path+'.aff')
  tokenizer = RegexpTokenizer(r'\w+')

  stem_col = []
  for text in tqdm.tqdm(ds.X):
    stems = []
    tokens = tokenizer.tokenize(text)
    for tok in tokens:
      tok_stems = hs.stem(tok)
      if len(tok_stems) > 0:
	      stems.append(tok_stems[0].decode())
    stem_col.append(' '.join(stems))

  ds.X = stem_col

def oversample(ds, target_f=np.average):
    
    class_counts = np.sum(ds.y, axis=0)
    class_avg = target_f(class_counts)

    rates_per_class = np.copy(class_counts).astype(float)
    rates_per_class[rates_per_class > 0] = class_avg / class_counts[class_counts > 0]
    rates_per_class[(rates_per_class > 0) & (rates_per_class < 1.0)] = 1.0
    rates_per_class = rates_per_class.round().astype(int)

    rates_per_example = np.max(np.where(ds.y == 1, rates_per_class, 0), axis=1)
    y_oversampled = np.repeat(ds.y, rates_per_example, axis=0)
    X_oversampled = np.repeat(ds.X, rates_per_example, axis=0)
    ids_oversampled = np.repeat(ds.ids, rates_per_example, axis=0)

    shuffled_indices = np.random.permutation(len(X_oversampled))

    ds_new = DatasetSplit()
    ds_new.ids = ids_oversampled[shuffled_indices]
    ds_new.X = X_oversampled[shuffled_indices]
    ds_new.y = y_oversampled[shuffled_indices]
    
    return ds_new

def train_svm(ds, min_df=1, max_df=1.0, loss_f='squared_hinge', c=1.0, max_iter=1000, n_jobs=1):

    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    estimator = LinearSVC(loss=loss_f, max_iter=max_iter, C=c)
    estimator = CalibratedClassifierCV(estimator)
    estimator = OneVsRestClassifier(estimator, n_jobs=n_jobs)

    pipe = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('model', estimator),
    ])
    pipe.fit(ds.X, ds.y)

    return pipe

def evaluate_split(ds, text_classifier, threshold=None, top_n=None, beta=1):

    kwargs = {}
    if threshold is not None:
        kwargs['threshold'] = threshold
    if top_n is not None:
        kwargs['top_n'] = top_n
    y_pred = text_classifier.predict(ds.X, **kwargs)
    
    f = fbeta_score(ds.y, y_pred, beta=beta, average='micro')
    p = precision_score(ds.y, y_pred, average='micro')
    r = recall_score(ds.y, y_pred, average='micro')

    return f, p, r



# ID_COLUMN = 'id'
# TEXT_COLUMN = 'text'
# LABEL_SEP = '|'

# def load_dataset(train_tsv, test_tsv=None, dev_tsv=None, label_column=None):

#     data_files = {
#         'train': train_tsv,
#     }
#     if dev_tsv is not None: data_files['dev'] = dev_tsv
#     if test_tsv is not None: data_files['test'] = test_tsv

#     data = datasets.load_dataset('csv', delimiter='\t', data_files=data_files)

#     for name, ds in data.items():
#         if ID_COLUMN not in ds.features or TEXT_COLUMN not in ds.features:
#             raise KeyError(f"Column '{ID_COLUMN}' or '{TEXT_COLUMN}' not found in '{name}'")
#         if len(data['train'].features) < 3:
#             raise ValueError(f"TSV file must contain at least one column in addition to '{ID_COLUMN}' and '{TEXT_COLUMN}' in '{name}'")
    
#     if label_column is None:
#         label_columns = [ f for f in data['train'].features if f not in [ID_COLUMN, TEXT_COLUMN] ]
#         label_column = label_columns[0]

#     data = data.map(lambda e: {'labels': e[label_column].split(LABEL_SEP)})

#     for split_name in data.keys():
#         rm_columns = [ c for c in data['train'].features if c not in [ID_COLUMN, TEXT_COLUMN, 'labels'] ]
#         data = data.remove_columns(rm_columns)

#     return data
