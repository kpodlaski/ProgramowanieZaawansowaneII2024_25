import torch
import torch.nn as nn

class RNNNet(nn.Module):
    def __init__(self, type):
        super(RNNNet, self).__init__()
        if type == 'LSTM':
            self.rnn = nn.LSTM(input_size=4, hidden_size=50, num_layers=1, batch_first=True)
        if type == 'GRU':
            self.rnn = nn.GRU(input_size=4, hidden_size=50, num_layers=1, batch_first=True)
        self.output = nn.Linear(50,1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.output(x)
        return x

    def summary(self, size):
        print(self)