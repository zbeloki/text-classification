from classifier import Classifier

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
    
    def __init__(self, model, is_multilabel):
        super().__init__(model, is_multilabel)

    @staticmethod
    def load(path, is_multilabel):
        model = pipeline('text-classification', model=path, **PIPELINE_KWARGS)
        return TransformerClassifier(model, is_multilabel)
            
    @classmethod
    def train(cls, is_multilabel, train_split, dev_split=None, n_trials=0, model_id=None, *args, **kwargs):
        train_hf = train_split.to_hf()
        dev_hf = None if dev_split is None else dev_split.to_hf()

        MAX_LEN = 512
        tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=MAX_LEN)

        def tokenize(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, return_tensors='pt')
        train_hf = train_hf.map(tokenize, batched=True)
        if dev_hf is not None:
            dev_hf = dev_hf.map(tokenize, batched=True)
        
        def model_init():
            model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=train_split.n_classes)
            if is_multilabel:
                model.config.problem_type = "multi_label_classification"
            return model

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            metrics = cls._evaluate_logits(labels, logits, is_multilabel)
            return metrics

        

        with tempfile.TemporaryDirectory() as tmp_dir:
            default_params = cls._default_hyperparameters()
            args = TrainingArguments(tmp_dir,
                                     save_strategy="epoch",
                                     gradient_accumulation_steps=2,
                                     **default_params)

            callbacks = []
            if dev_hf is not None:
                # Keep best model
                args.load_best_model_at_end = True
                args.evaluation_strategy = 'epoch'
                args.metric_for_best_model = 'f'
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

            if n_trials > 0:
                compute_objective = lambda metrics: metrics['eval_f']
                best_run = trainer.hyperparameter_search(n_trials=n_trials,
                                                         direction='maximize',
                                                         hp_space=cls._sample_hyperparameters,
                                                         compute_objective=compute_objective)

        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        pipe = pipeline('text-classification',
                        model=trainer.model,
                        tokenizer=tokenizer,
                        **PIPELINE_KWARGS)

        return TransformerClassifier(pipe, is_multilabel)

    @staticmethod
    def _default_hyperparameters(trial=None):
        return {
            'num_train_epochs': 3,
            'learning_rate': 1e-5,
            'seed': 2,
            'per_device_train_batch_size': 16,
        }

    @staticmethod
    def _sample_hyperparameters(trial):
        return {
            'num_train_epochs': trial.suggest_int("num_train_epochs", 1, 6),
            'learning_rate': trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            'seed': trial.suggest_int("seed", 1, 40),
            'per_device_train_batch_size': trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        }

    @staticmethod
    def predict_probabilities(texts, model):
        if type(texts) == np.ndarray:
            texts = texts.tolist()
        pipeline_output = model(texts, truncation=True)
        return np.array([ [ e['score'] for e in doc_res ] for doc_res in pipeline_output ])

    def save(self, path):        
        self._model.save_pretrained(path)
