# Class that implements a MASH tower
import torch
import torch.nn as nn
from SMASH.Paragraph_encoder import ParagraphEncoder
from SMASH.Sentence_encoder import SentenceEncoder
from SMASH.Word_encoder import WordEncoder

class MashRNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.X1 = WordEncoder()
        self.X2 = SentenceEncoder()
        self.X3 = ParagraphEncoder()

    def forward(self,X):
        X1 = self.X1(X)   #(1,1024)
        X2 = self.X2(X)   #(1,1024)
        X3 = self.X3(X)   #(1,1024)
        mashRNN = torch.cat([X1,X2,X3], dim=1)   #(1, 3*1024)
        return mashRNN