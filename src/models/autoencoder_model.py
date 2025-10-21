"""
scRAG/src/models/autoencoder.py

Autoencoder for compressing high-dimensional scRNA-seq data into a latent space.
This latent representation can then be used with the GAN in gan_latent.py.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import numpy as np


class Autoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction of scRNA-seq data.
    
    Compresses high-dimensional gene expression data (e.g., 2000 HVGs or 50 PCs)
    into a low-dimensional latent space that can be used with GANs.
    
    Args:
        input_dim: Dimension of input data (e.g., number of HVGs or PCs)
        latent_dim: Dimension of latent space (should match GAN's latent_dim)
        hidden_dims: List of hidden layer dimensions for encoder/decoder
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: list = None,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Default architecture if not specified
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Build encoder
        encoder_layers = []
        last_dim = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(last_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout)
            ])
            last_dim = h_dim
        
        # Final layer to latent space
        encoder_layers.append(nn.Linear(last_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        last_dim = latent_dim
        
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(last_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout)
            ])
            last_dim = h_dim
        
        # Final layer back to input dimension
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input data to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed data of shape (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder and decoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstructed_x, latent_z)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def train_autoencoder(
    model: Autoencoder,
    data: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True
) -> list:
    """
    Train the autoencoder on scRNA-seq data.
    
    Args:
        model: Autoencoder model instance
        data: Training data as numpy array (n_cells, n_features)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print training progress
        
    Returns:
        List of losses per epoch
    """
    model = model.to(device)
    model.train()
    
    # Convert data to tensor
    data_tensor = torch.FloatTensor(data).to(device)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            
            # Forward pass
            x_recon, _ = model(batch_x)
            loss = criterion(x_recon, batch_x)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return losses


def get_latent_representations(
    model: Autoencoder,
    data: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu"
) -> np.ndarray:
    """
    Extract latent representations from trained autoencoder.
    
    Args:
        model: Trained autoencoder model
        data: Input data as numpy array (n_cells, n_features)
        batch_size: Batch size for inference
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Latent representations as numpy array (n_cells, latent_dim)
    """
    model = model.to(device)
    model.eval()
    
    data_tensor = torch.FloatTensor(data).to(device)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    
    latents = []
    
    with torch.no_grad():
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            z = model.encode(batch_x)
            latents.append(z.cpu().numpy())
    
    return np.vstack(latents)


def reconstruct_from_latent(
    model: Autoencoder,
    latent_vectors: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu"
) -> np.ndarray:
    """
    Reconstruct high-dimensional data from latent vectors.
    
    Args:
        model: Trained autoencoder model
        latent_vectors: Latent representations (n_samples, latent_dim)
        batch_size: Batch size for inference
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Reconstructed data (n_samples, input_dim)
    """
    model = model.to(device)
    model.eval()
    
    latent_tensor = torch.FloatTensor(latent_vectors).to(device)
    dataset = torch.utils.data.TensorDataset(latent_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    
    reconstructions = []
    
    with torch.no_grad():
        for (batch_z,) in dataloader:
            batch_z = batch_z.to(device)
            x_recon = model.decode(batch_z)
            reconstructions.append(x_recon.cpu().numpy())
    
    return np.vstack(reconstructions)
