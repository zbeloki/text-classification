import json
import os
import pdb

class Config:

    def __init__(self,
                 is_multilabel=None,
                 f_beta=1.0,
                 optim_avg='weighted',
                 top_k=None):
        self.is_multilabel = is_multilabel
        self.f_beta = f_beta
        self.optim_avg = optim_avg
        self.top_k = top_k
        
    @classmethod
    def from_args(cls, args):
        pass

    @classmethod
    def from_dict(cls, dict):
        config = cls()
        for key, val in dict.items():
            if hasattr(config, key):
                setattr(config, key, val)
            else:
                # @: warning
                pass
        return config

    def kwargs(self, names):
        kwargs = {}
        for name in names:
            if not hasattr(self, name):
                raise KeyError(f"This config object doesn't contain the following parameter: {name}")
            val = getattr(self, name)
            if val is not None:
                kwargs[name] = val
        return kwargs

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

