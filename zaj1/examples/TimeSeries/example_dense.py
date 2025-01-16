import math

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from common.pytorch.ml_wrapper import ML_Wrapper
from examples.TimeSeries.dense_model import DenseNet
from examples.TimeSeries.generate_dataset import generate_sinus_dataset, generate_dataset_element
import matplotlib.pyplot as plt

training_dataset_size = 50000

dataset, dataloader = generate_sinus_dataset(training_dataset_size, 4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

base_path = "../../../"
n_epochs = 100
batch_size_train = 64
learning_rate = 0.01
momentum = 0.5

network = DenseNet()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
ml = ML_Wrapper(network, optimizer, base_path, device)
ml.summary((1,4))
ml.train(n_epochs,dataloader, loss_fun=F.mse_loss)
real_y = []
predicted_y = []

for k in range(4,1000):
    ry= math.sin(2 * math.pi * k / 1000)
    real_y.append(ry)
    pattern, _ = generate_dataset_element(k, 4)
    pred_y = ml.network(Tensor(pattern)).detach().numpy()
    predicted_y.append(pred_y[0])
plt.plot(real_y)
plt.plot(predicted_y)
plt.show()
