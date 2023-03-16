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

def lemmatize(texts, dic_path):

  hs = hunspell.HunSpell(dic_path+'.dic', dic_path+'.aff')
  tokenizer = RegexpTokenizer(r'\w+')

  stem_col = []
  for text in tqdm.tqdm(texts):
    stems = []
    tokens = tokenizer.tokenize(text)
    for tok in tokens:
      tok_stems = hs.stem(tok)
      if len(tok_stems) > 0:
	      stems.append(tok_stems[0].decode())
    stem_col.append(' '.join(stems))

  return stem_col

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
