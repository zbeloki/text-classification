import pytest
import numpy as np
import pdb

class TestClassificationOutput:

    def test_probas_binary(self, imdb_svm_out):
        probas = imdb_svm_out.probas
        assert probas.shape == (20, 2)
        assert np.around(probas[0][0], decimals=5) == 0.08896
        assert np.around(probas[2][1], decimals=5) == 0.59485
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

    def test_probas_multiclass(self, clothes, clothes_svm_out):
        probas = clothes_svm_out.probas
        assert probas.shape == (50, 5)
        assert np.around(probas[0][0], decimals=5) == 0.00969
        assert np.around(probas[2][1], decimals=5) == 0.16875
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

    def test_probas_multilabel(self, toxic_svm_out):
        probas = toxic_svm_out.probas
        assert probas.shape == (20, 6)
        assert np.around(probas[0][0], decimals=5) == 0.17851
        assert np.around(probas[2][1], decimals=5) == 0.02712

    def test_classes(self, clothes_svm_out):
        classes = clothes_svm_out.classes
        assert classes[:4].tolist() == ['Dresses', 'Dresses', 'Tops', 'Tops']
        assert len(classes) == 50
        assert len(set(classes)) == 3

    def test_class_probas(self, clothes_svm_out):
        classes, probas = clothes_svm_out.class_probas
        assert classes[:4].tolist() == ['Dresses', 'Dresses', 'Tops', 'Tops']
        assert len(classes) == 50
        assert len(set(classes)) == 3
        assert probas.shape == (50,)
        assert np.around(probas[0], decimals=5) == 0.79132
        assert np.around(probas[11], decimals=5) == 0.58038

    def test_class_indices(self, clothes_svm_out):
        class_indices = clothes_svm_out.class_indices
        assert class_indices.shape == (50,)
        assert class_indices.tolist()[:10] == [1, 1, 4, 4, 1, 1, 4, 4, 4, 4]

    def test_class_index_probas(self, clothes_svm_out):
        class_indices, probas = clothes_svm_out.class_index_probas
        assert class_indices.shape == (50,)
        assert class_indices.tolist()[:10] == [1, 1, 4, 4, 1, 1, 4, 4, 4, 4]
        assert probas.shape == (50,)
        assert np.around(probas[0], decimals=5) == 0.79132
        assert np.around(probas[11], decimals=5) == 0.58038
