# Class to encode the word layer
import torch
import torch.nn as nn

class WordEncoder(nn.Module):
    def __init__(self,input_dim=1024, hidden_dim=1024, hidden_layers=2, output_dim=1): #input_dim= input dim at each time step = emb_size
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.upw = nn.Linear(hidden_dim,output_dim)  #1024x1
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.softmax=nn.Softmax(dim=1)

    def forward(self, X):  # padded elmo embed
        x1, x2 = (X.shape[0]*X.shape[1]*X.shape[2], X.shape[3])   #(4*10*50, 1024)
        X = X.reshape(1, x1,-1)                                 #(1, 4*10*50, 1024)
        out = X.float()
        upkji = self.tanh(self.fc(out))            #(bs,2000,1024)
        alpha = self.softmax(self.upw(upkji))   #(bs,2000,1)
        first_level_attention = torch.sum(alpha*out, dim=1,keepdim=True)        #(bs,1,1024)
        first_level_attention = first_level_attention.reshape(first_level_attention.shape[1], first_level_attention.shape[2])
        return first_level_attention
