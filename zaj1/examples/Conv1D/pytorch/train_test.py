'based on https://nextjournal.com/gkoehler/pytorch-mnist'
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


from examples.Conv1D.pytorch.cnn1d_model import Conv1DNet
from common.pytorch.ml_wrapper import ML_Wrapper
from examples.Conv1D.read_dataset import read_acc_data_one_hot, read_acc_data, read_acc_data_pytorch

print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
base_path = "../../../"
n_epochs = 100
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10




network = Conv1DNet()

optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
ml = ML_Wrapper(network, optimizer, base_path, device)
#ml.summary((1,128))

train_loader, test_loader = read_acc_data_pytorch(batch_size_train, batch_size_test, data_loader_kwargs)

print("Start training")
for epoch in range(1, n_epochs + 1):
  ml.train(epoch, train_loader, test_loader)
ml.save_model("1D_network")
ml.training_plot()

