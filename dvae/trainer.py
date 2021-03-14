# Author: Mikita Sazanovich

import os

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as tutils
from tqdm import tqdm

import dataset
import utils
from dvae import model


def compute_loss(x_recon, x, mu, logvar, recon_loss_name):
  if recon_loss_name == 'bce_loss':
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
  elif recon_loss_name == 'mse_loss':
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
  else:
    raise NotImplementedError(f'Unknown reconstruction loss name: {recon_loss_name}.')
  kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return recon_loss, kl_loss


class Trainer:
  def __init__(self, config, output_dir):
    self.config = config
    self.output_dir = output_dir
    self.writer = SummaryWriter(os.path.join(output_dir, 'summaries'))
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.max_iter = config['max_iter']

    self.dataloader = dataset.get_dataloader(config, num_workers=12, preprocessing_type='dvae')
    self.hidden_dim = config['hidden_dim']
    self.beta = config['beta']

    self.recon_loss = config['recon_loss']
    channels = self.dataloader.dataset[0].size(0)
    self.model = model.BetaVAE_H(self.hidden_dim, channels)
    self.model.to(self.device)
    self.optim = optim.Adam(self.model.parameters(), lr=float(config['lr']))

  def train(self):
    self.model.train()
    progress_bar = tqdm(range(1, self.max_iter + 1))
    dataiter = iter(self.dataloader)
    for global_iter in progress_bar:
      try:
        imgs = next(dataiter)
      except StopIteration:
        dataiter = iter(self.dataloader)
        imgs = next(dataiter)
      imgs = imgs.to(self.device)
      imgs_recon, c, mu, logvar = self.model(imgs)

      recon_loss, kl_loss = compute_loss(imgs_recon, imgs, mu, logvar, self.recon_loss)
      loss = recon_loss + self.beta * kl_loss

      self.optim.zero_grad()
      loss.backward()
      self.optim.step()

      progress_bar.set_description(f'{global_iter}/{self.max_iter}: loss {loss:.0f}')

      if global_iter % 1000 == 0:
        self.writer.add_scalar('recon_loss', recon_loss, global_iter)
        self.writer.add_scalar('kl_loss', kl_loss, global_iter)
        imgs_grid = utils.compile_image_gallery(imgs, imgs_recon)
        tutils.save_image(imgs_grid, os.path.join(self.output_dir, 'samples', f'{global_iter}.png'))

      if global_iter % 100000 == 0 or global_iter == self.max_iter:
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'model_{global_iter}')
        self.model.save_state(checkpoint_path)
