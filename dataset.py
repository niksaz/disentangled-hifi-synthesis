# Author: Mikita Sazanovich

import os
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_dataloader(config, num_workers, preprocessing_type):
  name = config['dataset']
  batch_size = config['batch_size']
  image_size = config['image_size']
  if preprocessing_type == 'dvae':
    transform = transforms.Compose([
      transforms.Resize([image_size, image_size]),
      transforms.ToTensor(),
    ])
  elif preprocessing_type == 'idgan':
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
    ])
  else:
    raise ValueError(f'Unknown preprocessing type: {preprocessing_type}.')
  if name == 'dsprites':
    data_path = os.path.join('data', 'dsprites-dataset', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    if preprocessing_type == 'dvae':
      transform = None
    elif preprocessing_type == 'idgan':
      transform = lambda x: x * 2 - 1.0
    else:
      raise NotImplementedError
    dataset = get_non_tupled_tensor_dataset(data_path, transform)
  elif name == 'chairs':
    data_path = os.path.join('data', 'Chairs_64')
    dataset = get_non_tupled_image_folder_dataset(data_path, transform)
  elif name == 'cars':
    data_path = os.path.join('data', 'Cars_64')
    dataset = get_non_tupled_image_folder_dataset(data_path, transform)
  elif name == 'celeba':
    data_path = os.path.join('data', 'CelebA_64')
    dataset = get_non_tupled_image_folder_dataset(data_path, transform)
  elif name == 'celeba_128':
    data_path = os.path.join('data', 'CelebA_128')
    dataset = get_non_tupled_image_folder_dataset(data_path, transform)
  else:
    raise ValueError(f'Unknown dataset name: {name}.')
  dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
  return dataloader


def get_non_tupled_image_folder_dataset(data_path, transform):
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


def get_non_tupled_tensor_dataset(data_path, transform):
  data = np.load(data_path, encoding='bytes')
  data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
  dataset = NonTupledTensorDataset(data, transform)
  return dataset


class NonTupledTensorDataset(Dataset):
  def __init__(self, tensors, transform: Optional):
    self.tensors = tensors
    self.transform = transform

  def __getitem__(self, item):
    tensor = self.tensors[item]
    if self.transform is not None:
      tensor = self.transform(tensor)
    return tensor

  def __len__(self):
    return self.tensors.size(0)
