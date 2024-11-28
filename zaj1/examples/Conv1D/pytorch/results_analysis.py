import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


from examples.Conv1D.pytorch.cnn1d_model import Conv1DNet
from common.pytorch.ml_wrapper import ML_Wrapper
from examples.Conv1D.read_dataset import read_acc_data, read_acc_data_pytorch

base_path = "../../../"
n_epochs = 100
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


network = Conv1DNet()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
ml = ML_Wrapper.load_model (base_path, "1D_network_model.pth", network, optimizer, device)

train_loader, test_loader = read_acc_data_pytorch(batch_size_train, batch_size_test, data_loader_kwargs)

print("Test on training set:")
cf_matrix = ml.test(train_loader, create_confusion=True)
print(cf_matrix)
print("accuracy:", cf_matrix.trace()/cf_matrix.sum())
print("Test on test set:")
cf_matrix = ml.test(test_loader, create_confusion=True)
print(cf_matrix)
print("accuracy:", cf_matrix.trace()/cf_matrix.sum())

