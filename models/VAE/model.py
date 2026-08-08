"""
Phase: Model Definition & ELBO Loss
Expects: num_features, hidden_dim, latent_dim, kl_weight
Outputs: VAE PyTorch nn.Module class and vae_loss calculation function
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    Simple feedforward VAE over flow feature vectors.
    Input:  (batch, num_features)
    Output: reconstruction (batch, num_features), mu, logvar
    """
    def __init__(self, num_features, hidden_dim, latent_dim):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, kl_weight):
    """
    ELBO = reconstruction loss + KL divergence.
    """
    recon_loss = torch.mean((recon_x - x) ** 2, dim=1)                      # per-sample MSE
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # per-sample KL

    total_per_sample = recon_loss + kl_weight * kl_div
    return total_per_sample.mean(), recon_loss, kl_div
