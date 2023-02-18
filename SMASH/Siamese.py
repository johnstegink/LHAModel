# Class implement the Siamese netwerk containing two MASHRnn "towers"
import torch
import torch.nn as nn
from SMASH.MASH_rnn import MashRNN

class Siamese(nn.Module):   #Siamese (Unsupervised, we don't have labels)
    def __init__(self, preprocessor, dim=768):
        super().__init__()
        self.mashRNN = MashRNN()
        # self.linear = nn.Sequential(nn.Linear((preprocessor.max_sections + 1)* dim * 2, dim), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, x1, x2):
        out1 = self.mashRNN(x1)
        out2 = self.mashRNN(x2)
        cat =  torch.cat([out1,out2], dim=0)
        out = self.linear(cat)
        out = self.out(out)  # Zit la in de loss functie
        return out
