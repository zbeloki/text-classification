import numpy as np

from functools import cached_property
import pdb

class ClassificationOutput:

    def __init__(self, probas, label_binarizer, is_multilabel, threshold=0.5, top_k=None):
        self._probas = probas
        self._lb = label_binarizer
        self._is_multilabel = is_multilabel
        self._threshold = threshold if top_k is None else 0.0
        self._top_k = top_k

    @cached_property
    def probas(self):
        return self._probas

    @cached_property
    def y(self):
        if self._is_multilabel:
            return self._y_multilabel()
        else:
            return self._y_multiclass()

    @cached_property
    def classes(self):
        return self._lb.inverse_transform(self.y)

    @cached_property
    def class_probas(self):
        probas = self.probas[self.y.astype(bool)]
        return self.classes, probas

    @cached_property
    def class_indices(self):
        return np.argmax(self.y, axis=1)

    @cached_property
    def class_index_probas(self):
        probas = self.probas[self.y.astype(bool)]
        return self.class_indices, probas

    def _y_multilabel(self):
        y = np.copy(self._probas)
        y[y < self._threshold] = 0
        if self._top_k is not None:
            threshold_probas = -np.sort(-y)[:, self._top_k]
            threshold_probas = threshold_probas[..., np.newaxis]
            y[y <= threshold_probas] = 0
        y[y > 0] = 1
        return y.astype(int)

    def _y_multiclass(self):
        y = np.copy(self.probas)
        rows = np.arange(len(y))
        max_cols = np.argmax(y, axis=1)
        y[rows, max_cols] = 1
        y[y < 1] = 0
        return y.astype(int)

