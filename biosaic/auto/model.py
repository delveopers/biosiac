import torch
import torch.nn as nn
import math
from dataclasses import dataclass

@dataclass
class VQConfig:
  vocab_size: int = 256  # Size of DNA tokenizer vocab (e.g., 4^4 for 4-mer continuous)
  d_model: int = 512
  codebook_size: int = 1024  # Number of discrete codes in VQ codebook
  beta: float = 0.25
  n_heads: int = 8
  n_layers: int = 8
  dropout: float = 0.1
  max_seq_len: int = 1024

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_seq_len: int = 1024):
    super().__init__()
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
  def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, 
               dropout: float = 0.1, max_seq_len: int = 1024):
    super().__init__()
    self.embed = nn.Linear(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, 
      nhead=n_heads, 
      dropout=dropout,
      batch_first=True
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (B, L, vocab_size) - one-hot encoded or embedded input
    x = self.embed(x)  # (B, L, d_model)
    x = self.pos_encoding(x)
    x = self.dropout(x)
    z_e = self.encoder(x)  # (B, L, d_model)
    return z_e

class Decoder(nn.Module):
  def __init__(self, d_model: int, vocab_size: int, n_layers: int, n_heads: int, 
               dropout: float = 0.1, max_seq_len: int = 1024):
    super().__init__()
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
    decoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, 
      nhead=n_heads, 
      dropout=dropout,
      batch_first=True
    )
    self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
    self.fc_out = nn.Linear(d_model, vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, z_q: torch.Tensor) -> torch.Tensor:
    # z_q: (B, L, d_model)
    z_q = self.pos_encoding(z_q)
    z_q = self.dropout(z_q)
    x_recon = self.decoder(z_q)  # (B, L, d_model)
    x_recon = self.fc_out(x_recon)  # (B, L, vocab_size)
    return x_recon

class VectorQuantizer(nn.Module):
  def __init__(self, d_model: int, codebook_size: int, beta: float):
    super().__init__()
    self.d_model = d_model
    self.codebook_size = codebook_size
    self.beta = beta
    
    # Initialize codebook embeddings
    self.embeddings = nn.Embedding(codebook_size, d_model)
    self.embeddings.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

  def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # z_e: (B, L, d_model)
    B, L, D = z_e.shape
    z_e_flat = z_e.reshape(-1, self.d_model)  # (B*L, d_model)
    
    # Compute distances to codebook embeddings
    distances = torch.cdist(z_e_flat, self.embeddings.weight)  # (B*L, codebook_size)
    encoding_indices = torch.argmin(distances, dim=1)  # (B*L,)
    
    # Get quantized vectors
    z_q = self.embeddings(encoding_indices).view(B, L, D)  # (B, L, d_model)
    
    # VQ loss: commitment loss + embedding loss
    commitment_loss = self.beta * torch.mean((z_q.detach() - z_e) ** 2)
    embedding_loss = torch.mean((z_e.detach() - z_q) ** 2)
    vq_loss = commitment_loss + embedding_loss
    
    # Straight-through estimator
    z_q = z_e + (z_q - z_e).detach()
    
    return z_q, vq_loss, encoding_indices.view(B, L)

class DNA_VQVAE(nn.Module):
  def __init__(self, config: VQConfig):
    super().__init__()
    self.config = config
    
    self.encoder = Encoder(
      vocab_size=config.vocab_size,
      d_model=config.d_model,
      n_layers=config.n_layers,
      n_heads=config.n_heads,
      dropout=config.dropout,
      max_seq_len=config.max_seq_len
    )
    
    self.vq_layer = VectorQuantizer(
      d_model=config.d_model,
      codebook_size=config.codebook_size,
      beta=config.beta
    )
    
    self.decoder = Decoder(
      d_model=config.d_model,
      vocab_size=config.vocab_size,
      n_layers=config.n_layers,
      n_heads=config.n_heads,
      dropout=config.dropout,
      max_seq_len=config.max_seq_len
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: (B, L, vocab_size) - one-hot encoded DNA sequences
    z_e = self.encoder(x)  # (B, L, d_model)
    z_q, vq_loss, indices = self.vq_layer(z_e)  # (B, L, d_model), scalar, (B, L)
    x_recon = self.decoder(z_q)  # (B, L, vocab_size)
    return x_recon, vq_loss, indices

  def get_codebook_indices(self, x: torch.Tensor) -> torch.Tensor:
    """Get quantized indices for input sequences"""
    with torch.no_grad():
      z_e = self.encoder(x)
      _, _, indices = self.vq_layer(z_e)
    return indices

  def reconstruct_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
    """Reconstruct sequences from quantized indices"""
    with torch.no_grad():
      z_q = self.vq_layer.embeddings(indices)  # (B, L, d_model)
      x_recon = self.decoder(z_q)
    return x_recon