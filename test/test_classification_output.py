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

    def test_y_multiclass(self, clothes_svm_out):
        assert type(clothes_svm_out.y) == np.ndarray
        assert clothes_svm_out.y[0].tolist() == [0, 1, 0, 0, 0]
        assert clothes_svm_out.y[2].tolist() == [0, 0, 0, 0, 1]
        assert np.sum(clothes_svm_out.y) == 50

    def test_y_multilabel(self, toxic_svm_out, toxic_svm_out_top3, toxic_svm_out_th, toxic_svm_out_top3_th):
        assert type(toxic_svm_out.y) == np.ndarray
        assert np.sum(toxic_svm_out_top3.y) == 20 * 3
        # threshold: default
        assert toxic_svm_out.y[0].tolist() == [0, 1, 1, 0, 0, 1]
        assert toxic_svm_out_top3.y[0].tolist() == [0, 1, 1, 0, 0, 1]
        assert toxic_svm_out.y[2].tolist() == [0, 0, 0, 0, 0, 0]
        assert toxic_svm_out_top3.y[2].tolist() == [1, 0, 1, 1, 0, 0]
        assert toxic_svm_out.y[5].tolist() == [0, 1, 1, 1, 0, 1]
        assert toxic_svm_out_top3.y[5].tolist() == [0, 1, 1, 0, 0, 1]
        # threshold: 0.8
        assert toxic_svm_out_th.y[0].tolist() == [0, 1, 1, 0, 0, 1]
        assert toxic_svm_out_top3_th.y[0].tolist() == [0, 1, 1, 0, 0, 1]
        assert toxic_svm_out_th.y[1].tolist() == [0, 0, 0, 0, 0, 1]
        assert toxic_svm_out_top3_th.y[1].tolist() == [0, 0, 0, 0, 0, 1]
        assert toxic_svm_out_th.y[2].tolist() == [0, 0, 0, 0, 0, 0]
        assert toxic_svm_out_top3_th.y[2].tolist() == [0, 0, 0, 0, 0, 0]

    def test_labels_binary(self, imdb_svm_out, imdb_svm_out_lb):
        classes = imdb_svm_out_lb.labels
        assert type(classes) == list
        assert type(classes[0]) == str       
        assert classes[:4] == ['positive', 'positive', 'positive', 'negative']
        assert len(classes) == 20
        assert len(set(classes)) == 2
        classes = imdb_svm_out.labels
        assert type(classes) == list
        assert type(classes[0]) == int       
        assert classes[:4] == [1, 1, 1, 0]
        assert len(classes) == 20
        assert len(set(classes)) == 2

    def test_labels_multiclass(self, clothes_svm_out, clothes_svm_out_lb):
        classes = clothes_svm_out_lb.labels
        assert type(classes) == list
        assert type(classes[0]) == str        
        assert classes[:4] == ['Dresses', 'Dresses', 'Tops', 'Tops']
        assert len(classes) == 50
        assert len(set(classes)) == 3
        classes = clothes_svm_out.labels
        assert type(classes) == list
        assert type(classes[0]) == int        
        assert classes[:4] == [1, 1, 4, 4]
        assert len(classes) == 50
        assert len(set(classes)) == 3
        

    def test_labels_multilabel(self, toxic_svm_out, toxic_svm_out_lb):
        classes = toxic_svm_out_lb.labels
        assert type(classes) == list
        assert type(classes[0]) == list
        assert type(classes[0][0]) == str
        assert set(classes[0]) == set(['insult', 'obscene', 'toxic'])
        assert set(classes[4]) == set()
        assert set(classes[18]) == set(['insult', 'toxic'])
        assert len(classes) == 20
        classes = toxic_svm_out.labels
        assert type(classes) == list
        assert type(classes[0]) == list
        assert type(classes[0][0]) == int
        assert set(classes[0]) == set([1, 2, 5])
        assert set(classes[4]) == set()
        assert set(classes[18]) == set([1, 5])
        assert len(classes) == 20

    def test_label_probas_binary(self, imdb_svm_out, imdb_svm_out_lb):
        classes, probas = imdb_svm_out_lb.label_probas
        assert classes[:4] == ['positive', 'positive', 'positive', 'negative']
        assert len(classes) == 20
        assert type(probas) == list
        assert type(probas[0]) == float
        assert len(probas) == 20
        assert np.around(probas[0], decimals=3) == 0.911
        assert np.around(probas[2], decimals=3) == 0.595
        classes, probas = imdb_svm_out.label_probas
        assert classes[:4] == [1, 1, 1, 0]
        assert len(classes) == 20
        assert type(probas) == list
        assert type(probas[0]) == float
        assert len(probas) == 20
        assert np.around(probas[0], decimals=3) == 0.911
        assert np.around(probas[2], decimals=3) == 0.595

    def test_label_probas_multiclass(self, clothes_svm_out, clothes_svm_out_lb):
        classes, probas = clothes_svm_out_lb.label_probas
        assert classes[:4] == ['Dresses', 'Dresses', 'Tops', 'Tops']
        assert len(classes) == 50
        assert type(probas) == list
        assert type(probas[0]) == float
        assert len(probas) == 50
        assert np.around(probas[0], decimals=3) == 0.791
        assert np.around(probas[11], decimals=3) == 0.580
        classes, probas = clothes_svm_out.label_probas
        assert classes[:4] == [1, 1, 4, 4]
        assert len(classes) == 50
        assert type(probas) == list
        assert type(probas[0]) == float
        assert len(probas) == 50
        assert np.around(probas[0], decimals=3) == 0.791
        assert np.around(probas[11], decimals=3) == 0.580

    def test_label_probas_multilabel(self, toxic_svm_out, toxic_svm_out_lb):
        classes, probas = toxic_svm_out_lb.label_probas
        assert set(classes[0]) == set(['insult', 'obscene', 'toxic'])
        assert set(classes[4]) == set()
        assert len(classes) == 20
        assert type(probas) == list
        assert type(probas[0]) == list
        assert type(probas[0][0]) == float
        assert len(probas) == 20
        assert np.around(probas[0][0], decimals=3) == 0.946
        assert np.around(probas[5][3], decimals=3) == 0.987
        classes, probas = toxic_svm_out.label_probas
        assert set(classes[0]) == set([1, 2, 5])
        assert set(classes[4]) == set()
        assert len(classes) == 20
        assert type(probas) == list
        assert type(probas[0]) == list
        assert type(probas[0][0]) == float
        assert len(probas) == 20
        assert np.around(probas[0][0], decimals=3) == 0.946
        assert np.around(probas[5][3], decimals=3) == 0.987
