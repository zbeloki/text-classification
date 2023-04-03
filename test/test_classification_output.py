import pytest
import numpy as np
import pdb

class TestClassificationOutput:

    def test_probas_binary(self, imdb_svm_out):
        probas = imdb_svm_out.probas
        assert probas.shape == (20, 2)
        assert np.around(probas[0][0], decimals=3) == 0.089
        assert np.around(probas[2][1], decimals=3) == 0.595
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

    def test_probas_multiclass(self, clothes, clothes_svm_out):
        probas = clothes_svm_out.probas
        assert probas.shape == (50, 5)
        assert np.around(probas[0][0], decimals=3) == 0.010
        assert np.around(probas[2][1], decimals=3) == 0.169
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

    def test_probas_multilabel(self, toxic_svm_out):
        probas = toxic_svm_out.probas
        assert probas.shape == (20, 6)
        assert np.around(probas[0][0], decimals=3) == 0.179
        assert np.around(probas[2][1], decimals=3) == 0.027

    def test_classes_binary(self, imdb_svm_out):
        classes = imdb_svm_out.classes
        assert type(classes) == list
        assert type(classes[0]) == str        
        assert classes[:4] == ['positive', 'positive', 'positive', 'negative']
        assert len(classes) == 20
        assert len(set(classes)) == 2

    def test_classes_multiclass(self, clothes_svm_out):
        classes = clothes_svm_out.classes
        assert type(classes) == list
        assert type(classes[0]) == str        
        assert classes[:4] == ['Dresses', 'Dresses', 'Tops', 'Tops']
        assert len(classes) == 50
        assert len(set(classes)) == 3

    def test_classes_multilabel(self, toxic_svm_out):
        classes = toxic_svm_out.classes
        assert type(classes) == list
        assert type(classes[0]) == list
        assert type(classes[0][0]) == str
        assert set(classes[0]) == set(['insult', 'obscene', 'toxic'])
        assert set(classes[4]) == set()
        assert set(classes[18]) == set(['insult', 'toxic'])
        assert len(classes) == 20

    def test_class_probas_binary(self, imdb_svm_out):
        classes, probas = imdb_svm_out.class_probas
        assert classes[:4] == ['positive', 'positive', 'positive', 'negative']
        assert len(classes) == 20
        assert type(probas) == list
        assert type(probas[0]) == float
        assert len(probas) == 20
        assert np.around(probas[0], decimals=3) == 0.911
        assert np.around(probas[2], decimals=3) == 0.595

    def test_class_probas_multiclass(self, clothes_svm_out):
        classes, probas = clothes_svm_out.class_probas
        assert classes[:4] == ['Dresses', 'Dresses', 'Tops', 'Tops']
        assert len(classes) == 50
        assert type(probas) == list
        assert type(probas[0]) == float
        assert len(probas) == 50
        assert np.around(probas[0], decimals=3) == 0.791
        assert np.around(probas[11], decimals=3) == 0.580

    def test_class_probas_multilabel(self, toxic_svm_out):
        classes, probas = toxic_svm_out.class_probas
        assert set(classes[0]) == set(['insult', 'obscene', 'toxic'])
        assert set(classes[4]) == set()
        assert len(classes) == 20
        assert type(probas) == list
        assert type(probas[0]) == list
        assert type(probas[0][0]) == float
        assert len(probas) == 20
        assert np.around(probas[0][0], decimals=3) == 0.946
        assert np.around(probas[5][3], decimals=3) == 0.987

    def test_class_indices_binary(self, imdb_svm_out):
        class_indices = imdb_svm_out.class_indices
        assert type(class_indices) == list
        assert type(class_indices[0]) == int
        assert len(class_indices) == 20
        assert class_indices[:5] == [1, 1, 1, 0, 0]
        assert set(class_indices) == set([0, 1])

    def test_class_indices_multiclass(self, clothes_svm_out):
        class_indices = clothes_svm_out.class_indices
        assert type(class_indices) == list
        assert type(class_indices[0]) == int
        assert len(class_indices) == 50
        assert class_indices[:10] == [1, 1, 4, 4, 1, 1, 4, 4, 4, 4]

    def test_class_indices_multilabel(self, toxic_svm_out):
        class_indices = toxic_svm_out.class_indices
        assert type(class_indices) == list
        assert type(class_indices[0]) == list
        assert type(class_indices[0][0]) == int
        assert len(class_indices) == 20
        assert class_indices[:3] == [[1, 2, 5], [1, 2, 5], []]

    def test_class_index_probas_binary(self, imdb_svm_out):
        class_indices, probas = imdb_svm_out.class_index_probas
        assert type(class_indices) == list
        assert type(class_indices[0]) == int
        assert len(class_indices) == 20
        assert class_indices[:5] == [1, 1, 1, 0, 0]
        assert type(probas) == list
        assert type(probas[0]) == float
        assert len(probas) == 20
        assert np.around(probas[0], decimals=3) == 0.911
        assert np.around(probas[2], decimals=3) == 0.595

    def test_class_index_probas_multiclass(self, clothes_svm_out):
        class_indices, probas = clothes_svm_out.class_index_probas
        assert type(class_indices) == list
        assert type(class_indices[0]) == int
        assert len(class_indices) == 50
        assert class_indices[:10] == [1, 1, 4, 4, 1, 1, 4, 4, 4, 4]
        assert type(probas) == list
        assert type(probas[0]) == float
        assert len(probas) == 50
        assert np.around(probas[0], decimals=3) == 0.791
        assert np.around(probas[11], decimals=3) == 0.580

    def test_class_index_probas_multilabel(self, toxic_svm_out):
        class_indices, probas = toxic_svm_out.class_index_probas
        assert type(class_indices) == list
        assert type(class_indices[0]) == list
        assert type(class_indices[0][0]) == int
        assert len(class_indices) == 20
        assert class_indices[:3] == [[1, 2, 5], [1, 2, 5], []]
        assert type(probas) == list
        assert type(probas[0]) == list
        assert type(probas[0][0]) == float
        assert len(probas) == 20
        assert np.around(probas[0][0], decimals=3) == 0.946
        assert np.around(probas[5][3], decimals=3) == 0.987
