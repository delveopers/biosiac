import torch
import torch.nn as nn
import math

class ModelConfig:
  d_model: int = 128
  in_dim: int = 4
  n_embed: int = 512
  beta: float = 0.25
  n_heads: int = 8
  n_layers: int = 8
  dropout: float = 0.1
  max_seq_len: int = 1024

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_len=1024):
    super().__init__()
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
  def __init__(self, _in, d_model, n_layers, n_heads, dropout=0.1, max_seq_len=1024):
    super().__init__()
    self.embed = nn.Linear(_in, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
    self.encoder = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model=d_model, 
        nhead=n_heads, 
        dropout=dropout,
        batch_first=True
      ),
      num_layers=n_layers
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # x: (B, L, _in)
    x = self.embed(x)
    x = self.pos_encoding(x)
    x = self.dropout(x)
    z_e = self.encoder(x)  # (B, L, d_model)
    return z_e

class Decoder(nn.Module):
  def __init__(self, d_model, _out, n_layers, n_heads, dropout=0.1, max_seq_len=1024):
    super().__init__()
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
    self.decoder = nn.TransformerEncoder(  # Using encoder for autoencoder-style reconstruction
      nn.TransformerEncoderLayer(
        d_model=d_model, 
        nhead=n_heads, 
        dropout=dropout,
        batch_first=True
      ),
      num_layers=n_layers
    )
    self.fc_out = nn.Linear(d_model, _out)
    self.dropout = nn.Dropout(dropout)

  def forward(self, z_q):
    # z_q: (B, L, d_model)
    z_q = self.pos_encoding(z_q)
    z_q = self.dropout(z_q)
    x_recon = self.decoder(z_q)  # (B, L, d_model)
    x_recon = self.fc_out(x_recon)  # (B, L, _out)
    return x_recon

class Quantizer(nn.Module):
  def __init__(self, n_embed, d_model, beta):
    super().__init__()
    self.n_embed, self.d_model, self.beta = n_embed, d_model, beta
    self.embeddings = nn.Embedding(n_embed, d_model)
    self.embeddings.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

  def forward(self, z_e):
    # z_e: (B, L, d_model)
    B, L, D = z_e.shape
    z_e_flat = z_e.reshape(-1, self.d_model)  # (B*L, d_model)
    
    # Compute distances to embeddings
    distances = torch.cdist(z_e_flat, self.embeddings.weight)  # (B*L, n_embed)
    encoding_indices = torch.argmin(distances, dim=1)  # (B*L,)
    
    # Get quantized vectors
    z_q = self.embeddings(encoding_indices).view(B, L, D)  # (B, L, d_model)
    
    # VQ loss: commitment loss + embedding loss
    commitment_loss = self.beta * torch.mean((z_q.detach() - z_e) ** 2)
    embedding_loss = torch.mean((z_e.detach() - z_q) ** 2)
    loss = commitment_loss + embedding_loss
    
    # Straight-through estimator
    z_q = z_e + (z_q - z_e).detach()
    
    return z_q, loss, encoding_indices.view(B, L)

class DNA_VQVAE(nn.Module):
  def __init__(self, args: ModelConfig):
    super().__init__()
    self.encoder = Encoder(
      args.in_dim, 
      args.d_model, 
      args.n_layers, 
      args.n_heads, 
      args.dropout, 
      args.max_seq_len
    )
    self.vq_layer = Quantizer(args.n_embed, args.d_model, args.beta)
    self.decoder = Decoder(
      args.d_model, 
      args.in_dim, 
      args.n_layers, 
      args.n_heads, 
      args.dropout, 
      args.max_seq_len
    )

  def forward(self, x):
    # x: (B, L, in_dim)
    z_e = self.encoder(x)  # (B, L, d_model)
    z_q, vq_loss, indices = self.vq_layer(z_e)  # (B, L, d_model), scalar, (B, L)
    x_recon = self.decoder(z_q)  # (B, L, in_dim)
    return x_recon, vq_loss, indices

  def encode(self, x):
    """Encode input to quantized latent codes"""
    z_e = self.encoder(x)
    _, _, indices = self.vq_layer(z_e)
    return indices

  def decode(self, indices):
    """Decode from quantized indices"""
    z_q = self.vq_layer.embeddings(indices)  # (B, L, d_model)
    return self.decoder(z_q)