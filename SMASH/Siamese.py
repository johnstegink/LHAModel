# Class implement the Siamese netwerk containing two MASHRnn "towers"
import torch
import torch.nn as nn
from SMASH.MASH_rnn import MashRNN

class Siamese(nn.Module):   #Siamese (Unsupervised, we don't have labels)
    def __init__(self):
        super().__init__()
        self.mashRNN = MashRNN()
        self.linear = nn.Sequential(nn.Linear(2*3072, 1024), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x1, x2):
        out1 = self.mashRNN(x1)
        out2 = self.mashRNN(x2)
        cat =  torch.cat([out1,out2], dim=1)
        out = self.linear(cat)
        out = self.out(out)
        return out
