import torch
import torch.nn as nn
from typing import *
from torch.serialization import add_safe_globals
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

def create_encoder_decoder(model_path: str, vocab_size: int) -> tuple[DNA_VQVAEEncoder, DNA_VQVAEDecoder]:
  """
  Create encoder and decoder instances from a trained VQ-VAE model checkpoint.

  Handles both old (pickled VQConfig) and new (dict-based) config formats safely.

  Args:
    model_path: Path to the checkpoint
    vocab_size: Vocabulary size used for training

  Returns:
    Tuple of (encoder, decoder)
  """

  try:
    add_safe_globals([VQConfig])      # Trust only if loading old checkpoint (pickle-based config)
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    raw_config = checkpoint['config']

    # Convert config if it's a dict (new style)
    if isinstance(raw_config, dict):
      config = VQConfig(**raw_config)
    else:
      config = raw_config  # old pickled VQConfig object

  except Exception as e:
    raise RuntimeError(f"Failed to load checkpoint config from {model_path}: {e}")

  # Create and load model
  vqvae = DNA_VQVAE(config, vocab_size)
  vqvae.load_state_dict(checkpoint['model_state_dict'])

  # Freeze and return encoder/decoder
  encoder = DNA_VQVAEEncoder(vqvae)
  decoder = DNA_VQVAEDecoder(vqvae)
  return encoder, decoder

# sequence = "ATTTGGGGGATTAGTTGGGCGAACGGGTGAGTAACACGTGGGCAATCTGCCCTGCACTCTGGGACAAGCCCTGGAAACGGGGTCTAATACCGGATATGACCACTAGGGGCATCCCTTGGTGGTGTAAAGCTCCGGCGGTGCAGGATGACCCCGCGGCCTATCACCTTGTTGGTGAGGTAACGGCTCACCAAGGCAACAACGGGTAGCCGGCCTGAAAGGGCAACCGGCCACACTGGGACTGAAACACGGCCCAAACTCC"
# tokenizer = bio.Tokenizer("dna", 4, True)

# dna_ids = tokenizer.encode(sequence)  # [123, 89, 201, ...] (0-255 range)
# one_hot = torch.nn.functional.one_hot(torch.tensor(dna_ids), num_classes=tokenizer._tokenizer.vocab_size).float()

# vqvae_encoder, vqvae_decoder = create_encoder_decoder(
#   model_path="/content/drive/MyDrive/checkpoints/best_model.pth",
#   vocab_size=custom_vocab_size
# )

# vq_indices = vqvae_encoder(one_hot)
# reconstructed_one_hot = vqvae_decoder(vq_indices)
# reconstructed_dna_ids = torch.argmax(reconstructed_one_hot, dim=-1)
# reconstructed_sequence = tokenizer.decode(reconstructed_dna_ids.tolist()[0])

# print(vq_indices)
# print(reconstructed_sequence)