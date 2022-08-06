import torch.nn as nn


class SingleLayerLinear(nn.Module):
    def __init__(self, dim):
        super(SingleLayerLinear, self).__init__()
        self.fc1 = nn.Linear(dim, 1)
        self.dim = dim

    def forward(self, x):
        x = self.fc1(x)
        return x
