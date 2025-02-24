
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def compute_loss(self, pred, target):
        pass
    
    def get_metrics(self):
        return ['rmse', 'abs_rel', 'log10', 'delta1', 'delta2', 'delta3']