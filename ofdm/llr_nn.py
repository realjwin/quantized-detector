import numpy as np

import torch
import torch.nn as nn

def weighted_mse(llr_est, llr, epsilon):
    return torch.mean((llr_est - llr)**2 / (torch.abs(llr) + epsilon))

class LLRestimator(nn.Module):
    def __init__(self, ofdm_size, bits_per_symbol, expansion):
        super(LLRestimator, self).__init__()
        
        self.ofdm_size = ofdm_size
        
        self.activation = nn.Tanh()

        self.hidden1 = nn.Linear(2*self.ofdm_size+1, expansion*bits_per_symbol*self.ofdm_size, bias=True)
        self.hidden2 = nn.Linear(expansion*bits_per_symbol*self.ofdm_size, expansion*bits_per_symbol*self.ofdm_size, bias=True)
        self.hidden3 = nn.Linear(expansion*bits_per_symbol*self.ofdm_size, expansion*bits_per_symbol*self.ofdm_size, bias=True)
        
        self.final = nn.Linear(expansion*bits_per_symbol*self.ofdm_size, bits_per_symbol*self.ofdm_size, bias=True)
        
    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        
        return self.activation(self.final(x))
    
class LLRestimator_channel(nn.Module):
    def __init__(self, ofdm_size, bits_per_symbol, expansion):
        super(LLRestimator_channel, self).__init__()
        
        self.ofdm_size = ofdm_size
        
        self.activation = nn.Tanh()

        self.hidden1 = nn.Linear(3*self.ofdm_size+1, expansion*bits_per_symbol*self.ofdm_size, bias=True)
        self.hidden2 = nn.Linear(expansion*bits_per_symbol*self.ofdm_size, expansion*bits_per_symbol*self.ofdm_size, bias=True)
        self.hidden3 = nn.Linear(expansion*bits_per_symbol*self.ofdm_size, expansion*bits_per_symbol*self.ofdm_size, bias=True)
        
        self.final = nn.Linear(expansion*bits_per_symbol*self.ofdm_size, bits_per_symbol*self.ofdm_size, bias=True)
        
    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        
        return self.activation(self.final(x))