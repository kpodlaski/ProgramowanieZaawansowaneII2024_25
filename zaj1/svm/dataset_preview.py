import torch
import torchvision
import matplotlib.pyplot as plt

#pip install torch, torchvision
base_path = "."
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'/datasets/', train=True, download=True),
    batch_size=1)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(base_path+'/datasets/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               ])),
    batch_size = 10)

test_set = enumerate(test_loader)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig =plt.figure()
for i in range(8):
      plt.subplot(2, 4, i + 1)
      plt.tight_layout()
      plt.imshow(example_data[i][0])
      plt.title("Prediction: {}")
      plt.xticks([])
      plt.yticks([])
fig.show()