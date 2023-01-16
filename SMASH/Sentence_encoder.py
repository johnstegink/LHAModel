# Class to encode the sentence layer
import torch
import torch.nn as nn

class SentenceEncoder(nn.Module):
    def __init__(self,input_dim=1024, hidden_dim=1024, hidden_layers=2, output_dim=1, keep_prob=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.GRU = nn.GRU(input_dim,int(hidden_dim/2), hidden_layers, batch_first=True, dropout=keep_prob, bidirectional=True)
        self.upw = nn.Linear(hidden_dim,output_dim)
        self.ups = nn.Linear(hidden_dim,output_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.softmax=nn.Softmax(dim=1)

    def forward(self, X):
        x1, x2, x3 = (X.shape[0]*X.shape[1], X.shape[2], X.shape[3])   # (4*10,50,1024)
        X = X.reshape(x1, x2, x3)
        out = X.float()
        upkj = self.tanh(self.fc(out))  # u(p)(kj) = (bs,40,50,1024)
        alpha=self.softmax(self.upw(upkj))
        first_level_attention = torch.sum(alpha*out, dim=-2).unsqueeze(0)                          #(bs,40,1024)
        out,_=self.GRU(first_level_attention)
        upk=self.tanh(self.fc1(out))
        alpha=self.softmax(self.ups(upk))
        second_level_attention=torch.sum(alpha*out,dim=-2)
        return second_level_attention
