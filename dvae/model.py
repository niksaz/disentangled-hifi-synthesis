# Based on the underlying Beta-VAE implementation from: https://github.com/1Konny/Beta-VAE/blob/master/model.py

import torch
import torch.nn as nn
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, channels, z_dim):
        super().__init__()
        self.channels = channels
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),    # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim),               # B, z_dim
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self, z_dim, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),                      # B, 256
            View((-1, 256, 1, 1)),                      # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),             # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),        # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),        # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),        # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),  # B, channels, 64, 64
        )

    def forward(self, x):
        x = self.model(x)
        return x


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, channels=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(channels, z_dim*2)
        self.decoder = Decoder(z_dim, channels)
        self.apply(lambda m: normal_init(m, mean=0.00, std=0.02))

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar

    def encode(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z).sigmoid()

    def save_state(self, path):
        checkpoint = {'encoder': self.encoder.state_dict(), 'decoder': self.decoder.state_dict()}
        with open(path, 'wb') as fout:
            torch.save(checkpoint, fout)

    def load_state(self, path):
        with open(path, 'rb') as fin:
            checkpoint = torch.load(fin)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])


def normal_init(m, mean, std):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
