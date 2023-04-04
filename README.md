# text-classification

Todo:
- Pass raw labels to sklearn instead of 2D indicator matrix, as it uses multilabel setting otherwise.
- Implement TransformerClassifier
- warning if inferring multilabel on trained as multiclass and viceversa. warning when training multilabel on multiclass dataset and viceversa.
- which metric to optimize? micro-f, macro-f, weighted-f... parametrizable?