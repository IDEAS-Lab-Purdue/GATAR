import torch
import torch.nn as nn


class Preprocessor(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Preprocessor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        raise NotImplementedError
    
    def create_cost_map(self, x):
        raise NotImplementedError
    
    def create_cost_map_batch(self, x):
        raise NotImplementedError
    
    def create_oc_map(self, x):
        raise NotImplementedError
    
    def create_oc_map_batch(self, x):
        raise NotImplementedError
    
    def create_target_map(self, x):
        raise NotImplementedError
    
    def create_target_map_batch(self, x):
        raise NotImplementedError


