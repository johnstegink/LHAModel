import torch.nn as nn

class NeuralNetworkStat(nn.Module):
    """
    Basic neural network, without masks
    """
    def __init__(self, N):
        super(NeuralNetworkStat, self).__init__()

        self.hidden1 = nn.Linear(N, N)
        self.dropout1 = nn.Dropout( 0.0)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(N, N)
        self.dropout2 = nn.Dropout( 0.0)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(N, 1)
        self.act_output = nn.Sigmoid()
    def forward(self, x):
        x1 = self.act1(self.hidden1(x))
        do1 = self.dropout1( x1)
        x2 = self.act2(self.hidden2(do1))
        do2 = self.dropout2( x2)
        x_out = self.act_output(self.output(do2))

        return x_out
