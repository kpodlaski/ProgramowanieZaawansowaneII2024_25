import queue

import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


from common.pytorch.ml_wrapper import ML_Wrapper
from examples.CNN.pytorch.cnn_model import ConvNet

activation = {}
def get_activation(name=None):
    def hook (model, input, output):
        activation[name]=(output.cpu().detach())
    return hook

base_path = "../../../"
n_epochs = 100
batch_size_train = 64
batch_size_test = 1
learning_rate = 0.01
momentum = 0.5
log_interval = 10
print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'/datasets/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True, **data_loader_kwargs)


network = ConvNet()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
ml = ML_Wrapper.load_model (base_path, "conv_network_model.pth", network, optimizer, device)

print("Sample results:")
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)



module = ml.network.conv1
hook = module.register_forward_hook(get_activation("conv1"))

with torch.no_grad():
  example_data =example_data.to(device)
  network.to(device)
  output = ml.network(example_data)
  print(module.weight.shape)
  print(activation['conv1'].shape)
  fig = plt.figure()
  weights = module.weight.cpu().detach()
  # Show filters of first conv layer
  for i in range(module.weight.shape[0]):
      plt.subplot(3, 4, i + 1)
      plt.tight_layout()
      plt.imshow(weights[i][0], cmap='gray', interpolation='none')
      plt.xticks([])
      plt.yticks([])
  fig.show()

  #activations of first conv layer
  image = example_data[0].cpu().detach()
  fig = plt.figure()
  for i in range(activation['conv1'].shape[1]):
      plt.subplot(3, 4, i + 1)
      plt.tight_layout()
      plt.imshow(activation['conv1'][0][i], cmap='gray', interpolation='none')
      plt.xticks([])
      plt.yticks([])
  ## original
  plt.subplot(3, 4, 12)
  plt.tight_layout()
  plt.imshow(image[0], cmap='gray', interpolation='none')
  plt.xticks([])
  plt.yticks([])
  fig.show()



