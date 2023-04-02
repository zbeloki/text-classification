import json
import os

class Config:

    def __init__(self, **kwargs):
        self.classification_type = kwargs.pop("classification_type", "multiclass")

    @property
    def is_multilabel(self):
        return self.classification_type == 'multilabel'
        
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    @classmethod
    def load(cls, path):
        fpath = os.path.join(path, 'config.json')
        with open(fpath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save(self, path):
        fpath = os.path.join(path, 'config.json')
        with open(fpath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
