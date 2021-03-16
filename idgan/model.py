# Author: Mikita Sazanovich

import numpy as np
import torch
import torch.nn as nn

import dvae.model


class DVAEGenerator(nn.Module):
  def __init__(self, z_dim, channels, image_size):
    super().__init__()
    assert image_size == 64
    self.model = dvae.model.Decoder(z_dim, channels)

  def forward(self, x):
    x = self.model(x)
    x = torch.tanh(x)
    return x

  def save_state(self, path):
    checkpoint = {'model': self.model.state_dict()}
    with open(path, 'wb') as fout:
      torch.save(checkpoint, fout)

  def load_state(self, path):
      with open(path, 'rb') as fin:
          checkpoint = torch.load(fin)
      self.model.load_state_dict(checkpoint['model'])


class DVAEDiscriminator(nn.Module):
  def __init__(self, channels, image_size):
    super().__init__()
    assert image_size == 64
    self.model = dvae.model.Encoder(channels, 1)

  def forward(self, x):
    x = self.model(x)
    return x

  def save_state(self, path):
    checkpoint = {'model': self.model.state_dict()}
    with open(path, 'wb') as fout:
      torch.save(checkpoint, fout)

  def load_state(self, path):
      with open(path, 'rb') as fin:
          checkpoint = torch.load(fin)
      self.model.load_state_dict(checkpoint['model'])


def get_filters_to_map_from(size_from, size_to, filters_init=64, filters_max=512):
  if size_from > size_to:
    size_from, size_to = size_to, size_from
    reverse_filters = False
  else:
    reverse_filters = True
  n_layers = int(np.log2(size_to / size_from))
  filters = [filters_init]
  for i in range(n_layers):
    filters.append(min(filters_max, filters[-1] * 2))
  if reverse_filters:
    filters.reverse()
  return filters


class ResNetBlock(nn.Module):
  def __init__(self, filters_in, filters_out):
    super().__init__()
    filters_hidden = min(filters_in, filters_out)
    layers = []
    layers.append(nn.LeakyReLU(negative_slope=2e-1))
    layers.append(nn.Conv2d(filters_in, filters_hidden, kernel_size=3, stride=1, padding=1))
    layers.append(nn.LeakyReLU(negative_slope=2e-1))
    layers.append(nn.Conv2d(filters_hidden, filters_out, kernel_size=3, stride=1, padding=1))
    self.model = nn.Sequential(*layers)
    self.model_alpha = 0.1
    if filters_in == filters_out:
      self.shortcut = nn.Identity()
    else:
      self.shortcut = nn.Conv2d(filters_in, filters_out, kernel_size=1, stride=1, padding=0, bias=False)

  def forward(self, x):
    return self.shortcut(x) + self.model_alpha * self.model(x)


class ResNetGenerator(nn.Module):
  def __init__(self, z_dim, channels, image_size):
    super().__init__()
    init_size = 4
    filters = get_filters_to_map_from(init_size, image_size)
    layers = []
    layers.append(nn.Linear(z_dim, filters[0] * init_size * init_size))
    layers.append(dvae.model.View((-1, filters[0], init_size, init_size)))
    for i in range(len(filters) - 1):
      layers.append(ResNetBlock(filters[i], filters[i + 1]))
      layers.append(nn.Upsample(scale_factor=2))
    layers.append(ResNetBlock(filters[-1], filters[-1]))
    layers.append(nn.LeakyReLU(negative_slope=2e-1))
    layers.append(nn.Conv2d(filters[-1], channels, kernel_size=3, stride=1, padding=1))
    layers.append(nn.Tanh())
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    x = self.model(x)
    return x

  def save_state(self, path):
    checkpoint = {'model': self.model.state_dict()}
    with open(path, 'wb') as fout:
      torch.save(checkpoint, fout)

  def load_state(self, path):
      with open(path, 'rb') as fin:
          checkpoint = torch.load(fin)
      self.model.load_state_dict(checkpoint['model'])


class ResNetDiscriminator(nn.Module):
  def __init__(self, channels, image_size):
    super().__init__()
    final_size = 4
    filters = get_filters_to_map_from(image_size, final_size)
    layers = []
    layers.append(nn.Conv2d(channels, filters[0], kernel_size=3, stride=1, padding=1))
    layers.append(ResNetBlock(filters[0], filters[0]))
    for i in range(len(filters) - 1):
      layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
      layers.append(ResNetBlock(filters[i], filters[i + 1]))
    layers.append(nn.LeakyReLU(negative_slope=2e-1))
    layers.append(dvae.model.View((-1, filters[-1] * final_size * final_size)))
    layers.append(nn.Linear(filters[-1] * final_size * final_size, 1))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    x = self.model(x)
    return x

  def save_state(self, path):
    checkpoint = {'model': self.model.state_dict()}
    with open(path, 'wb') as fout:
      torch.save(checkpoint, fout)

  def load_state(self, path):
      with open(path, 'rb') as fin:
          checkpoint = torch.load(fin)
      self.model.load_state_dict(checkpoint['model'])


GENERATORS = {
  'DVAEGenerator': DVAEGenerator,
  'ResNetGenerator': ResNetGenerator,
}

DISCRIMINATORS = {
  'DVAEDiscriminator': DVAEDiscriminator,
  'ResNetDiscriminator': ResNetDiscriminator,
}
