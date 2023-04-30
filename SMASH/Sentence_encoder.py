# Class to encode the sentence layer
import torch
import torch.nn as nn

class SentenceEncoder(nn.Module):
    def __init__(self,input_dim=768, hidden_dim=768, hidden_layers=2, output_dim=1, keep_prob=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
#        self.GRU = nn.GRU(input_dim,int(hidden_dim/2), hidden_layers, batch_first=True, dropout=keep_prob, bidirectional=True)
        self.GRU = nn.GRU(input_dim,int(hidden_dim/2), hidden_layers, batch_first=True, bidirectional=True)
        self.upw = nn.Linear(hidden_dim,output_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.softmax=nn.Softmax(dim=1)

    def forward(self, X):
        # Apply the bidirectional GRU
        shapes = X.shape
        x_copy = X.reshape((-1,)+shapes[2:])
        hidden, _ = self.GRU( x_copy)

        # The attention
        u_p_kj = self.tanh(self.fc(hidden))
        alpha=self.softmax(self.upw(u_p_kj))

        # Sum the sentences regarding the alpha for this paragraph
        p_p_k = torch.sum( alpha*hidden, dim=-2)

        p_p_k_copy = p_p_k.reshape( (shapes[0], shapes[1],) + p_p_k.shape[1:])
        return p_p_k_copy
