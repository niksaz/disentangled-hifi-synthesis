# Author: Mikita Sazanovich

import argparse
import os
import random

import numpy as np
import torch
from torchvision import utils as tutils
import yaml


def parse_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('config_path', type=str, help='Path to the config')
  parser.add_argument('tag', type=str, help='Name of the experiment')
  parser.add_argument('--seed', default=27, type=int, help='Randomness seed')
  return parser.parse_args(args)


def fix_random_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def prepare_output_dir(output_dir, config):
  if os.path.exists(output_dir):
    raise ValueError(f'Directory {output_dir} already exists. Can not use it for the output.')
  os.makedirs(output_dir)
  checkpoints_dir = os.path.join(output_dir, 'checkpoints')
  os.makedirs(checkpoints_dir)
  samples_dir = os.path.join(output_dir, 'samples')
  os.makedirs(samples_dir)
  summaries_dir = os.path.join(output_dir, 'summaries')
  os.makedirs(summaries_dir)
  config_path = os.path.join(output_dir, 'config.yaml')
  dump_config(config_path, config)
  return output_dir


def load_config(path):
  with open(path, 'r') as stream:
    doc = yaml.load(stream)
    return doc['config']


def dump_config(path, config):
  doc = {'config': config}
  with open(path, 'w') as stream:
    yaml.dump(doc, stream)


def compile_image_gallery(imgs1, imgs2):
  nrow = int(np.ceil(np.sqrt(imgs1.size(0))))
  imgs1_grid = tutils.make_grid(imgs1, nrow=nrow, padding=1, pad_value=1)
  imgs2_grid = tutils.make_grid(imgs2, nrow=nrow, padding=1, pad_value=1)
  imgs_grid = tutils.make_grid(torch.stack([imgs1_grid, imgs2_grid]), nrow=2, padding=10, pad_value=0)
  return imgs_grid


def save_state(module, path):
  checkpoint = {'state_dict': module.state_dict()}
  with open(path, 'wb') as fout:
    torch.save(checkpoint, fout)


def load_state(module, path):
  with open(path, 'rb') as fin:
    checkpoint = torch.load(fin)
  module.load_state_dict(checkpoint['state_dict'])
