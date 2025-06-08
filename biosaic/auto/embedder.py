import torch
import torch.nn as nn

class ModelConfig:
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  A = 4        # DNA alphabet
  C = 21       # 21 letter for amino acid & 4 for dna
  d_msa = 128
  d_pair = 64
  n_heads = 8
  n_blocks = 4
  dropout = 0.2

class RowAttention(nn.Module):
  def __init__(self, d_msa, n_heads, dropout):
    super().__init__()
    self.attn = nn.MultiheadAttention(d_msa, n_heads, batch_first=True, dropout=dropout)
    self.norm = nn.LayerNorm(d_msa)
    self.dropout = nn.Dropout(dropout)

  def forward(self, msa):  # msa: (B, N, L, d_msa)
    B, N, L, D = msa.shape
    # Reshape: treat each position across sequences as a batch
    x = msa.transpose(1, 2).reshape(B * L, N, D)  # (B*L, N, d_msa)

    # Self-attention across sequences at each position
    attn_out, _ = self.attn(x, x, x)  # (B*L, N, d_msa)
    attn_out = self.dropout(attn_out)

    # Reshape back and apply residual connection
    attn_out = attn_out.view(B, L, N, D).transpose(1, 2)  # (B, N, L, d_msa)
    return self.norm(msa + attn_out)

class ColAttention(nn.Module):
  def __init__(self, d_msa, n_heads, dropout):
    super().__init__()
    self.attn = nn.MultiheadAttention(d_msa, n_heads, batch_first=True, dropout=dropout)
    self.norm = nn.LayerNorm(d_msa)
    self.dropout = nn.Dropout(dropout)

  def forward(self, msa):  # msa: (B, N, L, d_msa)
    B, N, L, D = msa.shape
    # Reshape: treat each sequence across positions as a batch
    x = msa.reshape(B * N, L, D)  # (B*N, L, d_msa)
    
    # Self-attention across positions for each sequence
    attn_out, _ = self.attn(x, x, x)  # (B*N, L, d_msa)
    attn_out = self.dropout(attn_out)
    
    # Reshape back and apply residual connection
    attn_out = attn_out.view(B, N, L, D)  # (B, N, L, d_msa)
    return self.norm(msa + attn_out)

class TriMulUpdate(nn.Module):
  def __init__(self, d_pair):
    super().__init__()
    self.linear_a = nn.Linear(d_pair, d_pair)
    self.linear_b = nn.Linear(d_pair, d_pair)
    self.gate = nn.Linear(d_pair, d_pair)
    self.norm = nn.LayerNorm(d_pair)
    self.activation = nn.GELU()

  def forward(self, pair):
    # pair: (B, L, L, d_pair)
    B, L, _, D = pair.shape
    
    # Apply linear transformations
    left = self.activation(self.linear_a(pair))   # (B, L, L, d_pair)
    right = self.activation(self.linear_b(pair))  # (B, L, L, d_pair)
    
    # Triangular multiplicative update
    # Compute: new_pair[i,j] += sum_k left[i,k] * right[k,j]
    update = torch.matmul(left, right.transpose(-2, -1))  # (B, L, L, d_pair)
    
    # Gating mechanism
    gate = torch.sigmoid(self.gate(pair))
    update = gate * update
    
    # Residual connection and normalization
    return self.norm(pair + update)

class MSARowAttentionWithPairBias(nn.Module):
  def __init__(self, d_msa, d_pair, n_heads, dropout):
    super().__init__()
    self.d_msa = d_msa
    self.n_heads = n_heads
    self.head_dim = d_msa // n_heads
    
    self.q_proj = nn.Linear(d_msa, d_msa)
    self.k_proj = nn.Linear(d_msa, d_msa)
    self.v_proj = nn.Linear(d_msa, d_msa)
    self.pair_bias = nn.Linear(d_pair, n_heads)
    self.out_proj = nn.Linear(d_msa, d_msa)
    
    self.dropout = nn.Dropout(dropout)
    self.norm = nn.LayerNorm(d_msa)

  def forward(self, msa, pair):
    # msa: (B, N, L, d_msa), pair: (B, L, L, d_pair)
    B, N, L, D = msa.shape
    
    # Compute attention with pair bias
    q = self.q_proj(msa).view(B, N, L, self.n_heads, self.head_dim)
    k = self.k_proj(msa).view(B, N, L, self.n_heads, self.head_dim)
    v = self.v_proj(msa).view(B, N, L, self.n_heads, self.head_dim)
    
    # Attention scores
    scores = torch.einsum('bnihd,bnjhd->bnhij', q, k) / (self.head_dim ** 0.5)
    
    # Add pair bias
    pair_bias = self.pair_bias(pair).permute(0, 3, 1, 2)  # (B, n_heads, L, L)
    scores = scores + pair_bias.unsqueeze(1)  # (B, N, n_heads, L, L)
    
    # Apply attention
    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = self.dropout(attn_weights)
    
    attn_out = torch.einsum('bnhij,bnjhd->bnihd', attn_weights, v)
    attn_out = attn_out.reshape(B, N, L, D)
    attn_out = self.out_proj(attn_out)
    
    return self.norm(msa + self.dropout(attn_out))

class Block(nn.Module):
  def __init__(self, d_msa, d_pair, n_heads, dropout):
    super().__init__()
    self.row_attn = MSARowAttentionWithPairBias(d_msa, d_pair, n_heads, dropout)
    self.col_attn = ColAttention(d_msa, n_heads, dropout)
    self.tri_mul = TriMulUpdate(d_pair)
    
    # Feedforward networks
    self.msa_ff = nn.Sequential(
      nn.Linear(d_msa, 4 * d_msa),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(4 * d_msa, d_msa),
      nn.Dropout(dropout)
    )
    self.pair_ff = nn.Sequential(
      nn.Linear(d_pair, 4 * d_pair),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(4 * d_pair, d_pair),
      nn.Dropout(dropout)
    )
    
    self.msa_norm = nn.LayerNorm(d_msa)
    self.pair_norm = nn.LayerNorm(d_pair)

  def forward(self, msa, pair):
    # MSA processing
    msa = self.row_attn(msa, pair)
    msa = self.col_attn(msa)
    msa = self.msa_norm(msa + self.msa_ff(msa))
    
    # Pair processing
    pair = self.tri_mul(pair)
    pair = self.pair_norm(pair + self.pair_ff(pair))
    
    return msa, pair

class Evoformer(nn.Module):
  def __init__(self, params: ModelConfig):
    """
      A: alphabet size (e.g. 4 for DNA, 21 for protein)
      C: number of initial pair features
    """
    super().__init__()
    self.embed_msa = nn.Linear(params.A, params.d_msa)
    self.embed_pair = nn.Linear(params.C, params.d_pair)
    
    self.blocks = nn.ModuleList([
      Block(params.d_msa, params.d_pair, params.n_heads, params.dropout)
      for _ in range(params.n_blocks)
    ])
    
    # Output heads
    self.msa_out = nn.Linear(params.d_msa, params.A)
    self.pair_out = nn.Linear(params.d_pair, params.C)
    
    # Dropout
    self.dropout = nn.Dropout(params.dropout)

  def forward(self, msa, pair):
    # msa: (B, N, L, A); pair: (B, L, L, C)
    msa = self.dropout(self.embed_msa(msa))    # (B, N, L, d_msa)
    pair = self.dropout(self.embed_pair(pair))  # (B, L, L, d_pair)
    
    # Apply Evoformer blocks
    for block in self.blocks:
      msa, pair = block(msa, pair)
    
    # Output projections
    msa_logits = self.msa_out(msa)   # (B, N, L, A)
    pair_logits = self.pair_out(pair)  # (B, L, L, C)
    
    return msa_logits, pair_logits