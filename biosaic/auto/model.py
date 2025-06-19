import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class VQConfig:
  d_model: int = 1536
  codebook_size: int = 2048
  beta: float = 0.25
  gamma: float = 0.99  # EMA decay for codebook updates
  n_heads: int = 16
  n_layers: int = 10
  dropout: float = 0.2
  max_seq_len: int = 1024
  label_smoothing: float = 0.1

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
  def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 1024):
    super().__init__()
    self.embed = nn.Linear(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

    # Add layer normalization and residual connections
    self.layers = nn.ModuleList([
      nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        batch_first=True,
        norm_first=True  # Pre-norm for better training stability
      ) for _ in range(n_layers)
    ])

    self.final_norm = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = self.embed(x)
    x = self.pos_encoding(x)
    x = self.dropout(x)

    for layer in self.layers:
      x = layer(x, src_key_padding_mask=mask)

    return self.final_norm(x)

class Decoder(nn.Module):
  def __init__(self, d_model: int, vocab_size: int, n_layers: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 1024):
    super().__init__()
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

    # Use proper decoder layers
    self.layers = nn.ModuleList([
      nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        batch_first=True,
        norm_first=True
      ) for _ in range(n_layers)
    ])

    self.final_norm = nn.LayerNorm(d_model)
    self.fc_out = nn.Linear(d_model, vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, z_q: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    z_q = self.pos_encoding(z_q)
    z_q = self.dropout(z_q)

    for layer in self.layers:
      z_q = layer(z_q, memory, tgt_mask=tgt_mask)

    z_q = self.final_norm(z_q)
    return self.fc_out(z_q)

class VectorQuantizer(nn.Module):
  def __init__(self, d_model: int, codebook_size: int, beta: float, gamma: float = 0.99):
    super().__init__()
    self.d_model = d_model
    self.codebook_size = codebook_size
    self.beta = beta
    self.gamma = gamma

    # Initialize codebook with better initialization
    self.embeddings = nn.Embedding(codebook_size, d_model)
    nn.init.xavier_uniform_(self.embeddings.weight)

    # EMA for codebook updates
    self.register_buffer('cluster_size', torch.zeros(codebook_size))
    self.register_buffer('embed_avg', self.embeddings.weight.data.clone())

    # Track codebook usage
    self.register_buffer('codebook_used', torch.zeros(codebook_size, dtype=torch.bool))

  def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, D = z_e.shape
    z_e_flat = z_e.reshape(-1, D)

    # Compute distances
    distances = torch.cdist(z_e_flat, self.embeddings.weight)
    encoding_indices = torch.argmin(distances, dim=1)

    # Update codebook usage tracking
    unique_indices = torch.unique(encoding_indices)
    self.codebook_used[unique_indices] = True

    # One-hot encoding for EMA updates
    encodings = F.one_hot(encoding_indices, self.codebook_size).float()
    # Get quantized vectors
    z_q = self.embeddings(encoding_indices).view(B, L, D)

    # EMA update of codebook (only during training)
    if self.training:
      self.cluster_size.data.mul_(self.gamma).add_(encodings.sum(0), alpha=1 - self.gamma)
      embed_sum = torch.matmul(encodings.t(), z_e_flat)
      self.embed_avg.data.mul_(self.gamma).add_(embed_sum, alpha=1 - self.gamma)

      # Normalize embeddings
      n = self.cluster_size.sum()
      cluster_size = (self.cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
      embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
      self.embeddings.weight.data.copy_(embed_normalized)

    # VQ loss with perplexity regularization
    commitment_loss = self.beta * F.mse_loss(z_q.detach(), z_e)
    embedding_loss = F.mse_loss(z_e.detach(), z_q)

    # Add perplexity regularization to encourage codebook usage
    avg_probs = encodings.mean(0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    perplexity_loss = -0.01 * perplexity  # Encourage higher perplexity

    vq_loss = commitment_loss + embedding_loss + perplexity_loss
    z_q = z_e + (z_q - z_e).detach()    # Straight-through estimator
    return z_q, vq_loss, encoding_indices.view(B, L)

  def get_codebook_usage(self) -> float:
    """Return percentage of codebook entries used"""
    return self.codebook_used.float().mean().item()

class DNA_VQVAE(nn.Module):
  def __init__(self, config: VQConfig, vocab_size: int):
    super().__init__()
    self.config = config
    self.vocab_size = vocab_size
    self.encoder = Encoder(vocab_size=vocab_size, d_model=config.d_model, n_layers=config.n_layers, n_heads=config.n_heads, dropout=config.dropout, max_seq_len=config.max_seq_len)    
    self.vq_layer = VectorQuantizer(d_model=config.d_model, codebook_size=config.codebook_size, beta=config.beta, gamma=config.gamma)
    self.decoder = Decoder(d_model=config.d_model, vocab_size=vocab_size, n_layers=config.n_layers, n_heads=config.n_heads, dropout=config.dropout, max_seq_len=config.max_seq_len)

    # Add reconstruction loss with label smoothing
    self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

  def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Encode
    z_e = self.encoder(x, mask)
    # Quantize
    z_q, vq_loss, indices = self.vq_layer(z_e)
    # Decode (using z_e as memory for cross-attention)
    x_recon = self.decoder(z_q, z_e)
    
    return x_recon, vq_loss, indices

  def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, vq_loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute reconstruction and total loss"""
    # Convert one-hot to class indices for CrossEntropyLoss
    if x.dim() == 3 and x.size(-1) == self.vocab_size:
      x_target = torch.argmax(x, dim=-1)
    else:
      x_target = x

    recon_loss = self.criterion(x_recon.reshape(-1, self.vocab_size), x_target.reshape(-1))
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss

  def get_codebook_indices(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Get quantized indices for input sequences"""
    with torch.no_grad():
      z_e = self.encoder(x, mask)
      _, _, indices = self.vq_layer(z_e)
    return indices

  def reconstruct_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
    """Reconstruct sequences from quantized indices"""
    with torch.no_grad():
      z_q = self.vq_layer.embeddings(indices)
      # Create dummy memory for decoder
      memory = z_q  # Use quantized vectors as memory
      x_recon = self.decoder(z_q, memory)
    return x_recon

  def get_codebook_usage(self) -> float:
    """Get codebook usage statistics"""
    return self.vq_layer.get_codebook_usage()