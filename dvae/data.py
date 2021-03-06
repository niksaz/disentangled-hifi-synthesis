# Author: Mikita Sazanovich

import os

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader


def get_dataloader(config):
  name = config['dataset']
  batch_size = config['batch_size']
  image_size = config['image_size']
  if name == 'dsprites':
    data_path = os.path.join('data', 'dsprites-dataset', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    data = np.load(data_path, encoding='bytes')
    data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
    dataset = NonTupledTensorDataset(data)
  else:
    raise ValueError(f'Unknown dataset name: {name}.')
  dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
  return dataloader


class NonTupledTensorDataset(Dataset):
  def __init__(self, tensors):
    self.tensors = tensors

  def __getitem__(self, item):
    return self.tensors[item]

  def __len__(self):
    return self.tensors.size(0)
