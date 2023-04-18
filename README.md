# text-classification

Config:

SVM:
classifier_config:
- type: "SVM"
- top_k
- classification_type
train_config:
- n_jobs
- max_iter
- optim_avg
- min_df
- max_df
- loss
- c

TRANSFORMER:
classifier_config:
- type: "TRANSFORMER"
- top_k
- classification_type
train_config:
- num_epochs
- seed
- optim_avg
- batch_size
- eval_batch_size
- grad_acc
- learning_rate
- weight_decay
- max_seq_len
- patience


Todo:
- Return label names
- CLI script

- warning if inferring multilabel on trained as multiclass and viceversa. warning when training multilabel on multiclass dataset and viceversa.
- which metric to optimize? micro-f, macro-f, weighted-f... parametrizable?
- add logging
- Transformer: predict_probabilities: improve and refactor
- Transformer: dev, compute_metrics, evaluate_logits
- Transformer: hyperparameter_search
- SVM: test hp_space
- package, versions and make pip installable
- Notebook(s)