' based on https://nextjournal.com/gkoehler/pytorch-mnist'
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.input = nn.Flatten()
        self.l1 = nn.Linear(28*28, 320)
        self.l2 = nn.Linear(320, 240)
        self.l3 = nn.Linear(240, 120)
        self.output = nn.Linear(120,10)

    def forward(self, x):
        x = self.input(x)
        x = torch.tanh(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.log_softmax(self.output(x),dim  =1)
        return x

    def summary(self, size):
        print(self)
