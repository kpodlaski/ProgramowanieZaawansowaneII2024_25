import torch
from torch import nn

from examples.Autoencoders.pytorch.Decoder import Decoder
from examples.Autoencoders.pytorch.Encoder import Encoder


class AutoencoderNet(nn.Module):
    def __init__(self, encoder_dim):
        super(AutoencoderNet, self).__init__()
        self.encoder = Encoder(encoded_space_dim=encoder_dim,fc2_input_dim=128)
        self.decoder = Decoder(encoded_space_dim=encoder_dim,fc2_input_dim=128)
        self.params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]

    def parameters(self):
        return [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




