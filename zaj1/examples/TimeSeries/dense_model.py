import torch
import torch.nn as nn

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.l1 = nn.Linear(4, 6)
        self.l2 = nn.Linear(6, 6)
        self.l3 = nn.Linear(6, 6)
        self.output = nn.Linear(6,1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = self.output(x)
        return x

    def summary(self, size):
        print(self)