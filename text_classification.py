from sklearn.metrics import fbeta_score, precision_score, recall_score

import joblib

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
