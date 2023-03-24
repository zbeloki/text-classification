# text-classification

Todo:
- Remove multilabel signal from dataset and add y(multilabel:bool) instead. Always keep the labels in a 2D matrix and convert on the fly when requesting y(multilabel=False)
- Implement TransformerClassifier