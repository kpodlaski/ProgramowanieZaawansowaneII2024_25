import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.max1 =  nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.tanh(self.max1(self.conv1(x)))
        x = torch.tanh(self.max2(self.conv2(x)))
        x = x.view(-1, 320)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x,dim  =1)

