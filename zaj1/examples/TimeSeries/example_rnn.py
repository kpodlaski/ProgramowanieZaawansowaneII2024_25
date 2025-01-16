import math

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from common.pytorch.ml_wrapper import ML_Wrapper
from examples.TimeSeries.dense_model import DenseNet
from examples.TimeSeries.generate_dataset import generate_sinus_dataset, generate_dataset_element
import matplotlib.pyplot as plt

from examples.TimeSeries.rnn_model import RNNNet

training_dataset_size = 50000

dataset, dataloader = generate_sinus_dataset(training_dataset_size, 4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

base_path = "../../../"
n_epochs = 10
batch_size_train = 64
learning_rate = 0.01
momentum = 0.5
rnn_type = 'LSTM'
#rnn_type = 'GRU'

network = RNNNet(rnn_type)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
ml = ML_Wrapper(network, optimizer, base_path, device)
#ml.summary((1,4))
for epoch in range(n_epochs):
    ml.train(epoch,dataloader, loss_fun=F.mse_loss)
real_y = []
predicted_y = []

for k in range(4,1000):
    ry= math.sin(2 * math.pi * k / 1000)
    real_y.append(ry)
    pattern, _ = generate_dataset_element(k, 4)
    pattern = Tensor(pattern).reshape(-1, 4).to(device)
    pred_y = ml.network(pattern).item()
    predicted_y.append(pred_y)
plt.plot(real_y)
plt.title(rnn_type + ' Sin(x) prediction')
plt.plot(predicted_y)
plt.show()
