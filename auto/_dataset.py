import torch
import torch.nn.functional as F
from typing import *
import os
import random
import biosaic as bio

class Dataset:
  """
  Initialize the Dataset
  Args:
    path (str): Path to the DNA data file
    kmer (int): kmer size for the tokenizer & encodings
    ratio (float): Fraction of data to use for validation (default 0.25)
    random_seed (int): random seeding for batching
    max_data_size (int): Maximum number of characters to load (default 100000)
  """
  def __init__(self, path: str, kmer: int, ratio: float = 0.25, random_seed: int = 1600, max_data_size: int = 100000):
    self.path = path
    self.ratio = max(0.0, min(1.0, ratio))  # Clamp ratio between 0 and 1
    self.random_seed = random_seed
    self.max_data_size = max_data_size
    self.kmer_size = max(1, kmer)  # Ensure kmer is at least 1

    try:
      self.tokenizer = bio.Tokenizer(mode="dna", kmer=self.kmer_size, continuous=True)
    except Exception as e:
      raise ValueError(f"Failed to initialize tokenizer: {e}")

    self.n_classes = self.tokenizer._tokenizer.vocab_size
    self.data = ""
    self.train_data = None
    self.val_data = None
    self._data_split = False
    self.load_and_format_data()

  def load_and_format_data(self):
    """
    Loads the file and formats the data:
      * Reads all lines
      * Strips whitespace and removes newline characters
      * Joins all lines into a single continuous string
      * Converts the string to uppercase
      * Filters out invalid DNA characters
    """
    if not os.path.isfile(self.path):
      raise FileNotFoundError(f"{self.path} does not exist.")

    try:
      with open(self.path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    except Exception as e:
      raise IOError(f"Failed to read file {self.path}: {e}")

    # Remove empty lines, strip whitespace, and join into one continuous string
    formatted_data = "".join(line.strip() for line in raw_lines if line.strip())

    # Convert to uppercase and filter valid DNA characters
    valid_chars = set('ACGT')
    filtered_data = ''.join(char for char in formatted_data.upper() if char in valid_chars)
    
    if not filtered_data:
      raise ValueError("No valid DNA sequences found in the file")

    # Limit data size
    self.data = filtered_data[:self.max_data_size]
    
    if len(self.data) < self.kmer_size:
      raise ValueError(f"Data length ({len(self.data)}) is less than k-mer size ({self.kmer_size})")

  def tokenize(self, seq: str) -> List[str]:
    """Tokenize a sequence into k-mers"""
    return self.tokenizer.tokenize(seq)

  def encode_seq(self, seq: str) -> List[int]:
    """Encode a sequence to token IDs"""
    return self.tokenizer.encode(seq)

  def decode_ids(self, ids: List[int]) -> str:
    """Decode token IDs back to sequence"""
    return self.tokenizer.decode(ids)

  def tokens_to_onehot(self, ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
    """Convert list of token IDs into one-hot encoded tensor of shape (N, vocab_size)"""
    if isinstance(ids, list):
      if not ids:
        return torch.empty(0, self.n_classes, dtype=torch.float)
      ids = torch.tensor(ids, dtype=torch.long)
    
    if ids.numel() == 0:
      return torch.empty(0, self.n_classes, dtype=torch.float)
    
    # Clamp IDs to valid range
    ids = torch.clamp(ids, 0, self.n_classes - 1)
    return F.one_hot(ids, num_classes=self.n_classes).float()  # shape (L, n_classes)

  def onehot_to_tokens(self, one_hot: torch.Tensor) -> List[int]:
    """Convert one-hot tensor back to list of token IDs"""
    if one_hot.numel() == 0:
      return []
    
    if one_hot.dim() != 2 or one_hot.size(1) != self.n_classes:
      raise ValueError(f"Expected one-hot of shape (N, {self.n_classes}), got {one_hot.shape}")
    
    return torch.argmax(one_hot, dim=-1).tolist()

  def train_test_split(self):
    """
    Splits the formatted data into training and validation sets
    Returns:
      A tuple (train_data, val_data) containing the split tensors
    """
    if not self.data:
      raise ValueError("Data is not loaded. Please check the file content.")
    
    if self._data_split:
      return self.train_data, self.val_data
    
    try:
      encoded_data = self.tokenizer.encode(self.data)
      if not encoded_data:
        raise ValueError("No valid tokens generated from data")
      
      encoded_tensor = self.tokens_to_onehot(encoded_data)
      
      if self.ratio == 0.0:
        self.train_data = encoded_tensor
        self.val_data = torch.empty(0, self.n_classes, dtype=torch.float)
      elif self.ratio == 1.0:
        self.train_data = torch.empty(0, self.n_classes, dtype=torch.float)
        self.val_data = encoded_tensor
      else:
        split_idx = int(len(encoded_tensor) * (1 - self.ratio))
        split_idx = max(1, min(split_idx, len(encoded_tensor) - 1))  # Ensure both splits have data
        
        self.train_data = encoded_tensor[:split_idx]
        self.val_data = encoded_tensor[split_idx:]
      
      self._data_split = True
      return self.train_data, self.val_data
      
    except Exception as e:
      raise RuntimeError(f"Failed to split data: {e}")

  def get_batch(self, split: str, batch_size: int, block_size: int, device: str = "cpu"):
    """
    Samples a random batch of subsequences from the train or validation data
    Args:
      split (str): "train" or "val"
      batch_size (int): Number of samples in the batch
      block_size (int): Length of each subsequence
      device (str): Device to move the tensors to (e.g. "cpu" or "cuda")
    Returns:
      Tuple of tensors (x, y) where x is the input batch and y is the target batch
      Note: For VQ-VAE, x and y are the same (autoencoder setup)
    """
    if split not in ["train", "val"]:
      raise ValueError("Split must be 'train' or 'val'")
    
    if batch_size <= 0 or block_size <= 0:
      raise ValueError("batch_size and block_size must be positive")
    
    train_data, val_data = self.train_test_split()
    data = train_data if split == "train" else val_data
    
    if data.numel() == 0:
      raise ValueError(f"No data available for split '{split}'")
    
    if len(data) < block_size:
      raise ValueError(f"Data length ({len(data)}) is less than block size ({block_size})")
    
    # Set random seed for reproducibility
    torch.manual_seed(self.random_seed)
    random.seed(self.random_seed)
    
    # Randomly choose starting indices
    max_start = len(data) - block_size
    if max_start <= 0:
      raise ValueError("Block size is too large for the available data")
    
    ix = torch.randint(0, max_start, (batch_size,))
    
    try:
      x = torch.stack([data[i:i+block_size] for i in ix])  # (B, L, n_classes)
      # For VQ-VAE (autoencoder), target is the same as input
      y = x.clone()
      
      return x.to(device), y.to(device)
      
    except Exception as e:
      raise RuntimeError(f"Failed to create batch: {e}")

  def get_full_data(self) -> str:
    """Returns the full formatted DNA string"""
    return self.data

  def get_data_stats(self) -> Dict[str, Any]:
    """Get statistics about the dataset"""
    train_data, val_data = self.train_test_split()
    return {
      'total_length': len(self.data),
      'train_tokens': len(train_data),
      'val_tokens': len(val_data),
      'vocab_size': self.n_classes,
      'kmer_size': self.kmer_size,
      'split_ratio': self.ratio
    }

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, idx: int) -> str:
    if idx < 0:
      idx = len(self.data) + idx  # Support negative indexing
    if idx < 0 or idx >= len(self.data):
      raise IndexError(f"Index {idx} out of range for data of length {len(self.data)}")
    return self.data[idx]