from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

import optuna
import pandas as pd
import numpy as np
import hunspell  # libhunspell-dev
from nltk.tokenize import RegexpTokenizer
import tqdm

import warnings
import re
import pdb

def clean_texts(texts):

  regex = re.compile('<.*?>')
  clean_texts = []
  for txt in tqdm.tqdm(texts):
    txt = re.sub(regex, ' ', txt)
    txt = ' '.join(txt.split())
    clean_texts.append(txt)
  return clean_texts

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

def oversample(X, y, target_f=np.average):

  ids = list(range(len(y)))
        
  class_counts = np.sum(y, axis=0)
  class_avg = target_f(class_counts)

  rates_per_class = np.copy(class_counts).astype(float)
  rates_per_class[rates_per_class > 0] = class_avg / class_counts[class_counts > 0]
  rates_per_class[(rates_per_class > 0) & (rates_per_class < 1.0)] = 1.0
  rates_per_class = rates_per_class.round().astype(int)

  rates_per_example = np.max(np.where(y == 1, rates_per_class, 0), axis=1)
  y_oversampled = np.repeat(y, rates_per_example, axis=0)
  X_oversampled = np.repeat(X, rates_per_example, axis=0)
  ids_oversampled = np.repeat(ids, rates_per_example, axis=0)
  
  shuf = np.random.permutation(len(ids_oversampled))

  return X_oversampled[shuf], y_oversampled[shuf], ids_oversampled[shuf]
