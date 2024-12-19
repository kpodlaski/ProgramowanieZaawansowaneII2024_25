'based on https://nextjournal.com/gkoehler/pytorch-mnist'
import os
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchsummary import summary

from examples.Autoencoders.pytorch.cynn_model import AutoencoderNet


class AutoEncoderWrapper():
    def __init__(self, network, optimizer, base_path, device, noise_factor=0.3):
        self.network=network
        self.noise_factor=noise_factor
        self.network.to(device)
        self.optimizer = optimizer
        self.device = device
        self.base_path = base_path
        self.train_losses = []
        self.val_losses = []

    def train(self, epoch, data_loader, val_loader = None):
        self.network.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data_plus_noise = self.add_noise(data)
            data = data.to(self.device)
            data_plus_noise = data_plus_noise.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data_plus_noise)
            loss = torch.nn.MSELoss()(output, data)
            train_loss+= loss.item()
            loss.backward()
            self.optimizer.step()
        train_loss = train_loss/len(data_loader.dataset)
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch, train_loss))
        self.train_losses.append(train_loss)

    def add_noise(self, inputs):
        noisy = inputs + torch.randn_like(inputs) * self.noise_factor
        noisy = torch.clip(noisy, 0., 1.)
        return noisy

    def save_model(self, fname):
        torch.save(self.network.state_dict(), self.base_path + '/out/{}_{}_model.pth'.format(fname, self.noise_factor))

    @staticmethod
    def get_model_path(base_path, file_name ):
        return base_path + "/out/" + file_name

    @staticmethod
    def load_model(base_path, file_name, model, optimizer=None, device = None ):
        file_path = AutoEncoderWrapper.get_model_path(base_path,file_name)
        model.load_state_dict(torch.load(file_path))
        return AutoEncoderWrapper(model, optimizer,base_path,device)


    def training_plot(self):
        fig = plt.figure()
        plt.plot([*range(len(self.train_losses))], self.train_losses, color='blue')
        if (len(self.val_losses))>0:
            plt.plot([*range(len(self.val_losses))], self.val_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('negative log likelihood loss')
        fig.show()

    def summary(self, size):
        x = torch.Tensor(size).to(self.device)
        summary(self.network, size)


print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
base_path = "../../../"
n_epochs = 30
batch_size_train = 256
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10
noise_factor = 0.2

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
code_dim = 4
network = AutoencoderNet(code_dim)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-05)

model_file_name = "autoencoder_model_noise.2.pth"


ml = AutoEncoderWrapper(network, optimizer, base_path, device)
ml.summary((1,28,28))

if os.path.exists(AutoEncoderWrapper.get_model_path(base_path,model_file_name)):
    print("Loading model from file:", model_file_name)
    ml = AutoEncoderWrapper.load_model(base_path, model_file_name, network, optimizer=optimizer, device=device)
else:
    print("Start training")
    for epoch in range(1, n_epochs + 1):
      ml.train(epoch, train_loader, test_loader)
    ml.save_model("autoencoder")



print("Sample results:")
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


with torch.no_grad():
  example_data =example_data.to(device)
  network.to(device)
  output = ml.network(example_data)

example_data =example_data.cpu()
output_data = output.data.cpu()
print(output_data.shape)
fig = plt.figure()
for i in range(6):
      plt.subplot(2, 6, i + 1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Original: ")
      plt.xticks([])
      plt.yticks([])
      plt.subplot(2, 6, i + 7)
      plt.imshow(output.data[i][0].cpu(), cmap='gray', interpolation='none')
      plt.title("Result:) ")
      plt.xticks([])
      plt.yticks([])
plt.tight_layout()
fig.show()
