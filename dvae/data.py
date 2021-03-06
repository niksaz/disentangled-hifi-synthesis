# Author: Mikita Sazanovich

import os

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_dataloader(config):
  name = config['dataset']
  batch_size = config['batch_size']
  image_size = config['image_size']
  if name == 'dsprites':
    data_path = os.path.join('data', 'dsprites-dataset', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataset = get_non_tupled_tensor_dataset(data_path)
  elif name == 'chairs':
    data_path = os.path.join('data', 'Chairs_64')
    dataset = get_non_tupled_image_folder_dataset(data_path, image_size)
  else:
    raise ValueError(f'Unknown dataset name: {name}.')
  dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
  return dataloader


def get_non_tupled_image_folder_dataset(data_path, image_size):
  transform = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
  ])
  image_folder = ImageFolder(root=data_path, transform=transform)
  dataset = NonTupledImageFolder(image_folder)
  return dataset


class NonTupledImageFolder(Dataset):
  def __init__(self, image_folder):
    self.image_folder = image_folder

  def __getitem__(self, item):
    return self.image_folder[item][0]

  def __len__(self):
    return len(self.image_folder)


def get_non_tupled_tensor_dataset(data_path):
  data = np.load(data_path, encoding='bytes')
  data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
  dataset = NonTupledTensorDataset(data)
  return dataset


class NonTupledTensorDataset(Dataset):
  def __init__(self, tensors):
    self.tensors = tensors

  def __getitem__(self, item):
    return self.tensors[item]

  def __len__(self):
    return self.tensors.size(0)
