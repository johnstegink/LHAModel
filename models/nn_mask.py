import torch.nn as nn

class NeuralNetworkMask(nn.Module):
    """
    Basic neural network, with mask added
    """
    def __init__(self, N):
        super(NeuralNetworkMask, self).__init__()
        vector_len = 2*N*N

        self.linear = nn.Sequential(
            nn.Linear(vector_len, int( vector_len / 2)),
            nn.Dropout( 0.2),
            nn.ReLU(),
            nn.Linear(int( vector_len / 2), int( vector_len / 10)),
            nn.ReLU(),
            nn.Linear(int( vector_len / 10), 1)
        )

    def forward(self, x):
        out = self.linear( x)
        return torch.sigmoid( out)
