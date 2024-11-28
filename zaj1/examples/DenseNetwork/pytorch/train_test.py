'based on https://nextjournal.com/gkoehler/pytorch-mnist'
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt


from examples.DenseNetwork.pytorch.dense_model import DenseNet
from common.pytorch.ml_wrapper import ML_Wrapper

print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
base_path = "../../../"
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'/datasets/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_train, shuffle=True, **data_loader_kwargs)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'/datasets/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True, **data_loader_kwargs)

network = DenseNet()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
ml = ML_Wrapper(network, optimizer, base_path, device)

print("Start training")
for epoch in range(1, n_epochs + 1):
  ml.train(epoch, train_loader, test_loader)
ml.save_model("dense_network")

print("Test on training set:")
ml.test(train_loader)
print("Test on test set:")
ml.test(test_loader)

ml.training_plot()


print("Sample results:")
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

with torch.no_grad():
  example_data =example_data.to(device)
  network.to(device)
  output = ml.network(example_data)

example_data =example_data.cpu()
fig = plt.figure()
for i in range(6):
      plt.subplot(2, 3, i + 1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Prediction: {}".format(
          output.data.max(1, keepdim=True)[1][i].item()))
      plt.xticks([])
      plt.yticks([])
fig.show()