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

class TransformerConfig(Config):

    def __init__(self,
                 model_id,
                 model_max_length=None,
                 num_train_epochs=None,
                 seed=None,
                 per_device_train_batch_size=None,
                 eval_batch_size=None,
                 gradient_accumulation_steps=None,
                 learning_rate=None,
                 weight_decay=None,
                 is_multilabel=None,
                 f_beta=1.0,
                 optim_avg='weighted',
                 top_k=None):
        super().__init__(is_multilabel, f_beta, optim_avg, top_k)
        self.model_id = model_id
        self.model_max_length = model_max_length
        self.num_train_epochs = num_train_epochs
        self.seed = seed
        self.per_device_train_batch_size = per_device_train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    @classmethod
    def from_args(cls, args):
        pass


class TransformerClassifier(Classifier):
    
    def __init__(self, config, label_binarizer, model, tokenizer):
        super().__init__(config, label_binarizer)
        self._tokenizer = tokenizer
        self._model = model
            
    @classmethod
    def train(cls, train_split, dev_split=None, config=None):
        # @: config default None? or force not None?
        
        config.type = 'Transformer'
        if config.is_multilabel is None:
            config.is_multilabel = train_split.is_multilabel

        label_binarizer = train_split.create_label_binarizer()
        train_hf = train_split.to_hf(label_binarizer)
        dev_hf = None if dev_split is None else dev_split.to_hf(label_binarizer)

        tokenizer_args = config.kwargs(['model_max_length'])
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, **tokenizer_args)

        def tokenize(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, return_tensors='pt')
        train_hf = train_hf.map(tokenize, batched=True)
        if dev_hf is not None:
            dev_hf = dev_hf.map(tokenize, batched=True)
        
        def model_init():
            model = AutoModelForSequenceClassification.from_pretrained(config.model_id, num_labels=train_split.n_classes)
            if config.is_multilabel:
                model.config.problem_type = "multi_label_classification"
            return model

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            eval_output = cls._evaluate_logits(labels, logits, is_multilabel)
            return {
                'f': eval_output.f(average=config.optim_avg),
                'precision': eval_output.precision(average=config.optim_avg),
                'recall': eval_output.recall(average=config.optim_avg),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir,
                                     save_strategy="epoch",
                                     optim='adamw_torch',
                                     **config.kwargs(['num_train_epochs',
                                                      'seed',
                                                      'per_device_train_batch_size',
                                                      'gradient_accumulation_steps']))

            callbacks = []
            if dev_hf is not None:
                # Keep best model
                args.load_best_model_at_end = True
                args.evaluation_strategy = 'epoch'
                args.metric_for_best_model = 'f'
                args.save_total_limit = 1
                # Early stopping
                #early_stop = EarlyStoppingCallback(early_stopping_patience=2,
                #                                   early_stopping_threshold=0.001)
                #callbacks.append(early_stop)
        
            trainer = Trainer(model_init=model_init,
                              args=args,
                              train_dataset=train_hf,
                              eval_dataset=dev_hf,
                              compute_metrics=compute_metrics,
                              callbacks=callbacks)

            trainer.train()
            
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
        device = get_device()
        tokens = self._tokenizer(texts, truncation=True, padding='longest', return_tensors='pt').to(device)
        output = self._model(**tokens)
        if self._config.is_multilabel:
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
