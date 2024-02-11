import torch
import torch.nn as nn

class NeuralNetworkPlain(nn.Module):
    """
    Basic neural network, without masks
    """
    def __init__(self, N):
        super(NeuralNetworkPlain, self).__init__()

        self.hidden1 = nn.Linear(N*N, 5)
        self.dropout1 = nn.Dropout(0.0)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(5, 1)
        self.act_output = nn.Sigmoid()

        torch.nn.init.xavier_uniform_( self.hidden1.weight)
        torch.nn.init.xavier_uniform_( self.output.weight)

    def forward(self, x):
        x1 = self.act1(self.hidden1(x))
        do1 = self.dropout1(x1)
        x_out = self.act_output(self.output(do1))

        return x_out
