import numpy as np

from functools import cached_property
import pdb

class ClassificationOutput:

    def __init__(self, probas, is_multilabel, label_binarizer=None, threshold=None, top_k=None):
        self._probas = probas
        self._lb = label_binarizer
        self._is_multilabel = is_multilabel
        self._threshold = threshold
        if threshold is None:
            self._threshold = 0.5 if top_k is None else 0.0
        self._top_k = top_k

    @cached_property
    def probas(self):
        return self._probas

    @cached_property
    def y(self):
        if self._is_multilabel:
            return self._y_2d_multilabel()
        else:
            return self._y_1d_multiclass()

    @cached_property
    def y_2d(self):
        if self._is_multilabel:
            return self._y_2d_multilabel()
        else:
            return self._y_2d_multiclass()

    @cached_property
    def labels(self):
        if self._lb is None:
            if self._is_multilabel:
                return [ np.where(y == 1)[0].tolist() for y in self.y_2d ]
            else:
                return self._y_1d_multiclass().tolist()
        else:
            classes = self._lb.inverse_transform(self.y_2d)
            if self._is_multilabel:
                return [ list(cls) for cls in classes ]
            else:
                return classes.tolist()            

    @cached_property
    def label_probas(self):
        if self._is_multilabel:
            probas = [ self.probas[i][self.y_2d[i]==1].tolist() for i in range(len(self.y_2d)) ]
        else:
            probas = self.probas[self.y_2d == 1].tolist()        
        return self.labels, probas

    
    # Helpers
    
    def _y_2d_multilabel(self):
        y = np.copy(self._probas)
        y[y < self._threshold] = 0
        if self._top_k is not None:
            threshold_probas = -np.sort(-y)[:, self._top_k-1]
            threshold_probas = threshold_probas[..., np.newaxis]
            y[y < threshold_probas] = 0
        y[y > 0] = 1
        return y.astype(int)

    def _y_1d_multiclass(self):
        return np.argmax(self._y_2d_multiclass(), axis=1)

    def _y_2d_multiclass(self):
        y = np.copy(self.probas)
        rows = np.arange(len(y))
        max_cols = np.argmax(y, axis=1)
        y[rows, max_cols] = 1
        y[y < 1] = 0
        return y.astype(int)
