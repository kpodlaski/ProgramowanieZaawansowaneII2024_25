import torch
import torchvision

from torchsummary import summary
import torch.nn as nn

model_conv = torchvision.models.alexnet()
model_conv = torchvision.models.resnet18()#weights='IMAGENET1K_V1')
#print(model_conv)
model_conv.cuda()
summary(model_conv, (3, 224, 224))