# Author: Mikita Sazanovich

import math
import os

import torch
from torch import autograd
from torch import distributions
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as tutils
from tqdm import tqdm

import dataset
import dvae.model
import utils
from idgan import model


def get_optimizer_for(parameters, optimizer, lr):
  if optimizer == 'sgd':
    return optim.SGD(parameters, lr=lr, momentum=0.0)
  elif optimizer == 'rmsprop':
    return optim.RMSprop(parameters, lr=lr, alpha=0.99, eps=1e-8)
  elif optimizer == 'adam':
    return optim.Adam(parameters, lr=lr, betas=(0.0, 0.99), eps=1e-8)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}.')


def compute_cross_entropy_loss_against(d_out, target):
  d_target = torch.full(size=d_out.size(), fill_value=target, dtype=d_out.dtype, device=d_out.device)
  loss = F.binary_cross_entropy_with_logits(d_out, d_target)
  return loss


def compute_id_loss(real_c, real_c_mu, real_c_logvar, fake_c, fake_c_mu, fake_c_logvar):
  loss = -(
      math.log(2*math.pi)
      + fake_c_logvar + (real_c - fake_c_mu).pow(2).div(fake_c_logvar.exp() + 1e-8)).div(2).sum(1).mean()
  return loss


def compute_gradient_penalty(outputs, inputs):
  grads = autograd.grad(
    outputs=outputs.sum(),
    inputs=inputs,
    create_graph=True,
    retain_graph=True,
    only_inputs=True
  )[0]
  grads2 = grads.pow(2)
  assert (grads2.size() == inputs.size())
  grads_penalty = grads2.view(grads2.size(0), -1).sum(1)
  return grads_penalty


class Trainer:
  def __init__(self, config, output_dir):
    self.config = config
    self.output_dir = output_dir
    self.writer = SummaryWriter(os.path.join(output_dir, 'summaries'))
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.max_iter = int(config['max_iter'])
    self.gp_param = float(config['gp_param'])
    self.info_alpha = float(config['info_alpha'])

    # Data
    self.batch_size = config['batch_size']
    self.dataloader = dataset.get_dataloader(config, num_workers=12, preprocessing_type='idgan')

    # Models
    self.c_dim = config['c_dim']
    self.s_dim = config['s_dim']
    channels = self.dataloader.dataset[0].size(0)
    self.dvae = dvae.model.BetaVAE_H(self.c_dim, channels)
    self.dvae.load_state(config['dvae_checkpoint'])
    z_dim = self.c_dim + self.s_dim
    generator_cls = model.GENERATORS[config['generator']['name']]
    generator_kwargs = {'z_dim': z_dim, 'channels': channels}
    self.generator = generator_cls(**generator_kwargs)
    discriminator_cls = model.DISCRIMINATORS[config['discriminator']['name']]
    discriminator_kwargs = {'channels': channels}
    self.discriminator = discriminator_cls(**discriminator_kwargs)

    self.dvae.to(self.device)
    self.generator.to(self.device)
    self.discriminator.to(self.device)

    # Latent distributions
    self.c_dist = distributions.Normal(torch.zeros(self.c_dim).to(self.device), torch.ones(self.c_dim).to(self.device))
    self.s_dist = distributions.Normal(torch.zeros(self.s_dim).to(self.device), torch.ones(self.s_dim).to(self.device))

    # Optimizers
    self.g_optimizer = get_optimizer_for(self.generator.parameters(), config['optimizer'], float(config['lr_g']))
    self.d_optimizer = get_optimizer_for(self.discriminator.parameters(), config['optimizer'], float(config['lr_d']))

  def train(self):
    self.dvae.train()
    self.generator.train()
    self.discriminator.train()
    progress_bar = tqdm(range(1, self.max_iter + 1))
    dataiter = iter(self.dataloader)
    for global_iter in progress_bar:
      # Sample a batch of real data
      try:
        x_real = next(dataiter)
      except StopIteration:
        dataiter = iter(self.dataloader)
        x_real = next(dataiter)
      x_real = x_real.to(self.device)

      # Extract the latent variable z for the real data
      with torch.no_grad():
        c, c_mu, c_logvar = self.dvae.encode((x_real + 1.0) / 2)
        s = self.s_dist.sample((self.batch_size,))
        z = torch.cat([s, c], 1)

      # Do a discriminator update
      dis_loss = self.discriminator_step(x_real, z)

      # Do a generator update
      gen_loss, x_fake = self.generator_step(z, c, c_mu, c_logvar)

      # Post-training
      progress_bar.set_description(f'{global_iter}/{self.max_iter}: dloss {dis_loss:.3f} gloss {gen_loss:.3f}')

      if global_iter % 1000 == 0:
        self.writer.add_scalar('dis_loss', dis_loss, global_iter)
        self.writer.add_scalar('gen_loss', gen_loss, global_iter)
        x_grid = utils.compile_image_gallery(x_real, x_fake)
        tutils.save_image(x_grid, os.path.join(self.output_dir, 'samples', f'{global_iter}.png'))

  def discriminator_step(self, x_real, z):
    self.dvae.zero_grad()
    self.discriminator.zero_grad()
    self.generator.zero_grad()

    x_real.requires_grad_()
    src_real = self.discriminator(x_real)
    loss_real = compute_cross_entropy_loss_against(src_real, 1.0)
    gp_loss = self.gp_param * compute_gradient_penalty(src_real, x_real).mean()
    with torch.no_grad():
      x_fake = self.generator(z)
    src_fake = self.discriminator(x_fake)
    loss_fake = compute_cross_entropy_loss_against(src_fake, 0.0)
    dis_loss = (loss_real + loss_fake) + self.gp_param * gp_loss
    dis_loss.backward()
    self.d_optimizer.step()

    return dis_loss.item()

  def generator_step(self, z, real_c, real_c_mu, real_c_logvar):
    self.dvae.zero_grad()
    self.discriminator.zero_grad()
    self.generator.zero_grad()

    x_fake = self.generator(z)
    src_fake = self.discriminator(x_fake)
    gen_gan_loss = compute_cross_entropy_loss_against(src_fake, 1.0)
    fake_c, fake_c_mu, fake_c_logvar = self.dvae.encode((x_fake + 1.0) / 2)
    id_loss = compute_id_loss(real_c, real_c_mu, real_c_logvar, fake_c, fake_c_mu, fake_c_logvar)
    gen_loss = gen_gan_loss - self.info_alpha * id_loss
    gen_loss.backward()
    self.g_optimizer.step()

    return gen_loss.item(), x_fake.detach()
