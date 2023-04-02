import numpy as np

class TestClassificationOutput:

    def test_probas(self, clothes_svm_out):
        probas = clothes_svm_out.probas
        assert probas.shape == (50, 5)
        assert np.around(probas[0][0], decimals=5) == 0.01110
        assert np.around(probas[2][1], decimals=5) == 0.15998
    
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
        assert np.around(probas[0], decimals=5) == 0.90624
        assert np.around(probas[11], decimals=5) == 0.71065

    def test_class_indices(self, clothes_svm_out):
        class_indices = clothes_svm_out.class_indices
        assert class_indices.shape == (50,)
        assert class_indices.tolist()[:10] == [1, 1, 4, 4, 1, 1, 4, 4, 4, 4]

    def test_class_index_probas(self, clothes_svm_out):
        class_indices, probas = clothes_svm_out.class_index_probas
        assert class_indices.shape == (50,)
        assert class_indices.tolist()[:10] == [1, 1, 4, 4, 1, 1, 4, 4, 4, 4]
        assert probas.shape == (50,)
        assert np.around(probas[0], decimals=5) == 0.90624
        assert np.around(probas[11], decimals=5) == 0.71065
