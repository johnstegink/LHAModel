# Class that implements a MASH tower
import torch
import torch.nn as nn
from SMASH.Paragraph_encoder import ParagraphEncoder
from SMASH.Sentence_encoder import SentenceEncoder
from SMASH.Word_encoder import WordEncoder

class MashRNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.X2 = SentenceEncoder()
        self.X3 = ParagraphEncoder()

    def forward(self,X):
        Sentences = self.X2(X)
        Paragraphs = self.X3(Sentences)



        mashRNN = torch.cat([torch.sum(Sentences, dim=-1),Paragraphs], dim=0)   #(1, 2*1024)
        # mashRNN = Paragraphs
        return mashRNN