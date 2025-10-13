"""
scRAG/src/models/gan_latent.py

Minimal GAN generator/discriminator for latent vectors.
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=32, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,1)
        )

    def forward(self, x):
        return self.net(x)
