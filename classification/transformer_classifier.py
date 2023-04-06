from .classifier import Classifier
from .config import Config

import torch
from transformers import TrainingArguments, AutoTokenizer, Trainer, EarlyStoppingCallback, AutoModelForSequenceClassification, pipeline
from datasets import Dataset

import numpy as np

import tempfile
import pdb

def get_device():
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        return f"cuda:{device_id}"
    else:
        return 'cpu'

PIPELINE_KWARGS = {
    'top_k': None,
    'device': get_device(),
}

class TransformerClassifier(Classifier):
    
    def __init__(self, config, label_binarizer, model, tokenizer):
        super().__init__(config, label_binarizer)
        self._pipe = pipeline('text-classification',
                              model=model,
                              tokenizer=tokenizer,
                              **PIPELINE_KWARGS)
            
    @classmethod
    def train(cls, train_split, dev_split=None, f_beta=1.0, top_k=None, model_id=None, *args, **kwargs):
        params = {
            'num_train_epochs': 1,
            'seed': 2,
            'per_device_train_batch_size': 16,
        }
        max_len = 512
        grad_acc = 2
        optim_metric = 'f'

        label_binarizer = train_split.create_label_binarizer()
        train_hf = train_split.to_hf(label_binarizer)
        dev_hf = None if dev_split is None else dev_split.to_hf(label_binarizer)

        tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=max_len)

        def tokenize(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, return_tensors='pt')
        train_hf = train_hf.map(tokenize, batched=True)
        if dev_hf is not None:
            dev_hf = dev_hf.map(tokenize, batched=True)
        
        def model_init():
            model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=train_split.n_classes)
            if train_split.is_multilabel:
                model.config.problem_type = "multi_label_classification"
            return model

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            eval_output = cls._evaluate_logits(labels, logits, is_multilabel)
            average = 'weighted'
            return {
                'f': eval_output.f(average),
                'precision': eval_output.precision(average),
                'recall': eval_output.recall(average),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir,
                                     save_strategy="epoch",
                                     gradient_accumulation_steps=grad_acc,
                                     optim='adamw_torch',
                                     **params)

            callbacks = []
            if dev_hf is not None:
                # Keep best model
                args.load_best_model_at_end = True
                args.evaluation_strategy = 'epoch'
                args.metric_for_best_model = optim_metric
                args.save_total_limit = 1
                # Early stopping
                early_stop = EarlyStoppingCallback(early_stopping_patience=2,
                                                   early_stopping_threshold=0.001)
                callbacks.append(early_stop)
        
            trainer = Trainer(model_init=model_init,
                              args=args,
                              train_dataset=train_hf,
                              eval_dataset=dev_hf,
                              compute_metrics=compute_metrics,
                              callbacks=callbacks)

            trainer.train()

        kwargs['top_k'] = top_k
        if 'classification_type' not in kwargs:
            kwargs['classification_type'] = 'multilabel' if train_split.is_multilabel else 'multiclass'
        config = Config.from_dict(kwargs)
            
        return TransformerClassifier(config, label_binarizer, trainer.model, tokenizer)

    @classmethod
    def search_hyperparameters(cls, train_split, dev_split, n_trials, f_beta=1, top_k=False):
        compute_objective = lambda metrics: metrics['eval_f']
        best_run = trainer.hyperparameter_search(n_trials=n_trials,
                                                 direction='maximize',
                                                 hp_space=cls._sample_hyperparameters,
                                                 compute_objective=compute_objective)

    @staticmethod
    def _sample_hyperparameters(trial):
        return {
            'num_train_epochs': trial.suggest_int("num_train_epochs", 1, 6),
            'learning_rate': trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            'seed': trial.suggest_int("seed", 1, 40),
            'per_device_train_batch_size': trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        }

    def predict_probabilities(self, texts):
        if type(texts) == np.ndarray:
            texts = texts.tolist()
        tokenizer = self._pipe.tokenizer
        model = self._pipe.model
        device = get_device()
        tokens = tokenizer(texts, truncation=True, padding='longest', return_tensors='pt').to(device)
        output = model(**tokens)
        if self._config.classification_type == 'multilabel':
            probas = torch.sigmoid(output.logits)
        else:
            probas = torch.nn.functional.softmax(output.logits, dim=1)
        return probas.detach().cpu().numpy()

    @classmethod
    def load(cls, path):
        model = pipeline('text-classification', model=path, **PIPELINE_KWARGS)
        return TransformerClassifier(model, is_multilabel)

    def save(self, path):        
        self._model.save_pretrained(path)
