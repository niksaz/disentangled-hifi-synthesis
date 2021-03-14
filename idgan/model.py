# Author: Mikita Sazanovich

import torch
import torch.nn as nn

import dvae.model


class DVAEGenerator(nn.Module):
  def __init__(self, z_dim, channels):
    super().__init__()
    self.model = dvae.model.Decoder(z_dim, channels)

  def forward(self, x):
    x = self.model(x)
    x = torch.tanh(x)
    return x


class DVAEDiscriminator(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.model = dvae.model.Encoder(channels, 1)

  def forward(self, x):
    x = self.model(x)
    return x
