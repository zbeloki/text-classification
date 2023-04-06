# text-classification

Todo:
- warning if inferring multilabel on trained as multiclass and viceversa. warning when training multilabel on multiclass dataset and viceversa.
- which metric to optimize? micro-f, macro-f, weighted-f... parametrizable?
- classifier::save: save label_binarizer and also load
- add logging
- Transformer: predict_probabilities: improve and refactor
- XXXClassifiers: Design better flow of arguments, parameters, config...
- Transformer: dev, compute_metrics, evaluate_logits