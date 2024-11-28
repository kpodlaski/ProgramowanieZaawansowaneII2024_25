import torch
import torch.nn as nn

class Conv1DNet(nn.Module):
    def __init__(self):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=10)
        #self.max1 =  nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv2 = nn.Conv1d(64, 64, kernel_size=10)
        #self.max2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.drop = nn.Dropout(.15)
        self.max1 = nn.MaxPool1d((2), stride=(2))
        self.fc1 = nn.Linear(64*55, 100)
        self.fc2 = nn.Linear(100, 6)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.drop(x)
        x = self.max1 (x)
        x = x = x.view(-1, 64*55)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x,dim  =1)

