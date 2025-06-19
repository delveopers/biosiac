import torch
import torch.nn as nn
from typing import *
from .model import DNA_VQVAE, VQConfig

class DNA_VQVAEEncoder(nn.Module):
  """Separate encoder class that inherits from the main model"""
  def __init__(self, vqvae_model: DNA_VQVAE):
    super().__init__()
    self.vqvae = vqvae_model
    # Freeze the model parameters for encoding-only usage
    for param in self.vqvae.parameters():
      param.requires_grad = False
    self.vqvae.eval()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Encode DNA sequences to discrete indices"""
    return self.vqvae.get_codebook_indices(x)

  def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
    """Alias for forward method"""
    return self.forward(x)

class DNA_VQVAEDecoder(nn.Module):
  """Separate decoder class that inherits from the main model"""
  def __init__(self, vqvae_model: DNA_VQVAE):
    super().__init__()
    self.vqvae = vqvae_model
    # Freeze the model parameters for decoding-only usage
    for param in self.vqvae.parameters():
      param.requires_grad = False
    self.vqvae.eval()

  def forward(self, indices: torch.Tensor) -> torch.Tensor:
    """Decode from discrete indices to DNA sequences"""
    return self.vqvae.reconstruct_from_indices(indices)

  def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
    """Alias for forward method"""
    return self.forward(indices)

# Utility function to create encoder/decoder from trained model
def create_encoder_decoder(model_path: str, config: Optional[VQConfig] = None) -> tuple[DNA_VQVAEEncoder, DNA_VQVAEDecoder]:
  """
  Create separate encoder and decoder instances from a trained VQ-VAE model
  
  Args:
    model_path: Path to the trained model checkpoint
    config: Model configuration (if not provided, will try to load from checkpoint)
  
  Returns:
    Tuple of (encoder, decoder) instances
  """
  if config is None:
    # Try to load config from checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'config' in checkpoint:
      config = checkpoint['config']
    else:
      raise ValueError("Config not found in checkpoint and not provided")
  
  # Create and load the main model
  vqvae = DNA_VQVAE(config)
  vqvae.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
  
  # Create separate encoder and decoder
  encoder = DNA_VQVAEEncoder(vqvae)
  decoder = DNA_VQVAEDecoder(vqvae)
  
  return encoder, decoder