from itertools import product
import json, pickle
import os, tempfile, urllib, requests
import numpy as np
import re

class DNA:
  def __init__(self, kmer: int, continuous: bool=True, special_tokens=None):
    self.kmer = kmer
    self.continuous = continuous
    self._base_chars = ['A', 'T', 'G', 'C', '-']
    self.vocab = {}

    # Handle special tokens
    self.special_tokens = special_tokens or ['<S>', '</S>', '<P>', '<C>', '<M>'] if special_tokens != False else []
    self.has_special_tokens = not self.continuous

    # Special tokens only work with continuous=False
    if self.has_special_tokens and continuous:
      raise ValueError("Special tokens are only supported with continuous=False")

    # Calculate base vocab size
    if self.continuous:
      self.vocab_size = len(self._base_chars) ** kmer
    else:
      self.vocab_size = sum(len(self._base_chars) ** i for i in range(1, kmer+1))
    if self.has_special_tokens:
      self.vocab_size += len(self.special_tokens)

  def _split_with_special_tokens(self, sequence):
    """Split sequence preserving special tokens"""
    if not self.has_special_tokens:
      return [sequence]

    pattern = '(' + '|'.join(re.escape(token) for token in self.special_tokens) + ')'
    parts = re.split(pattern, sequence)
    return [part for part in parts if part]  # Remove empty strings

  def tokenize(self, sequence):
    if not self.has_special_tokens:
      if any(ch not in self._base_chars for ch in sequence):
        raise ValueError("Invalid character in DNA sequence")
      return [sequence[i:i+self.kmer] for i in range(len(sequence) - self.kmer + 1)] if self.continuous else [sequence[i:i+self.kmer] for i in range(0, len(sequence), self.kmer)]

    tokens = []
    for part in self._split_with_special_tokens(sequence):
      if part in self.special_tokens:
        tokens.append(part)
      else:
        if any(ch not in self._base_chars for ch in part):
          raise ValueError("Invalid character in DNA sequence")
        tokens.extend([part[i:i+min(self.kmer, len(part)-i)] for i in range(0, len(part), self.kmer) if i < len(part)])
    return tokens

  def detokenize(self, ids):
    if self.continuous and not self.has_special_tokens:
      return "" if not ids else "".join(ids[i][0] for i in range(len(ids))) + ids[-1][1:]
    return "".join(ids)

  def build_vocab(self):
    letters, combos = sorted(self._base_chars), []
    if self.continuous:
      combos = list(product(letters, repeat=self.kmer))
    else:
      for L in range(1, self.kmer + 1):
        combos.extend(product(letters, repeat=L))

    self.vocab = {''.join(c): i for i, c in enumerate(combos)}
    if self.has_special_tokens:
      start_idx = len(self.vocab)
      for i, token in enumerate(self.special_tokens):
        self.vocab[token] = start_idx + i
    self.ids_to_token = {v: k for k, v in self.vocab.items()}
    self.vocab_size = len(self.vocab)

  def encode(self, sequence):
    sequence = sequence.upper()
    tokenized_data = self.tokenize(sequence)
    return [self.vocab[kmer] for kmer in tokenized_data if kmer in self.vocab]

  def decode(self, ids):
    return self.detokenize(self.ids_to_chars(ids))

  def ids_to_chars(self, ids: list[int]):
    assert isinstance(ids, list) and len(ids) > 0, "ids must be a non-empty list"
    assert isinstance(ids[0], int), "only accepts encoded ids"
    return [self.ids_to_token[i] for i in ids]

  def chars_to_ids(self, chars: list[str]):
    assert isinstance(chars, list) and len(chars) > 0, "chars must be a non-empty list"
    assert isinstance(chars[0], str), "only accepts tokenized strings"
    return [self.vocab[i] for i in chars]

  def verify(self, ids, file=None):
    verified = []
    ids = self.ids_to_chars(ids) if isinstance(ids[0], int) else ids
    for i in range(len(ids) - 1):
      # Skip verification for special tokens
      if ids[i] in self.special_tokens or ids[i+1] in self.special_tokens:
        verified.append({"kmer1": ids[i], "kmer2": ids[i + 1], "match": "special_token"})
        continue
      match = ids[i][1:] == ids[i + 1][:-1] if self.continuous else True
      verified.append({"kmer1": ids[i], "kmer2": ids[i + 1], "match": match})

    if file:
      with open(os.path.join(file, "verify.json"), 'w', encoding='utf-8') as f:
        json.dump(verified, f)
    return verified

  def save(self, path, as_json=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
      "kmer": self.kmer,
      "vocab_size": self.vocab_size,
      "trained_vocab": self.vocab,
      "special_tokens": self.special_tokens,
      "continuous": self.continuous
    }
    ext = ".json" if as_json else ".model"
    with open(path + ext, "w" if as_json else "wb", encoding="utf-8" if as_json else None) as f:
      json.dump(data, f, indent=2) if as_json else pickle.dump(data, f)
    print(f"DEBUGG INFO[104] [Saved] Vocabulary saved to {path + ext}")

  def load(self, model_path: str):
    def is_url(path):
      return path.startswith(("http://", "https://"))

    if is_url(model_path):
      with tempfile.NamedTemporaryFile(delete=False, suffix=".model" if model_path.endswith(".model") else ".json") as tmp_file:
        try:
          urllib.request.urlretrieve(model_path.replace("blob/", ""), tmp_file.name)
          model_path = tmp_file.name
        except Exception as e:
          raise RuntimeError(f"Failed to download model from {model_path}: {e}")

    with open(model_path, "r" if model_path.endswith(".json") else "rb", encoding="utf-8" if model_path.endswith(".json") else None) as f:
      data = json.load(f) if model_path.endswith(".json") else pickle.load(f)

    if not model_path.endswith((".json", ".model")):
      raise TypeError("Only supports vocab file format `.model` & `.json`")

    self.vocab = data["trained_vocab"]
    self.kmer = data.get("kmer", self.kmer)
    self.continuous = data.get("continuous", self.continuous)
    self.ids_to_token = {v: k for k, v in self.vocab.items()}
    self.has_special_tokens = not self.continuous
    if self.has_special_tokens:
      self.special_tokens = list(dict.fromkeys(data.get("special_tokens", [])))

      max_id = max(self.vocab.values(), default=-1)
      for token in self.special_tokens:
        if token not in self.vocab:
          max_id += 1
          self.vocab[token] = max_id
          self.ids_to_token[max_id] = token
    self.vocab_size = len(self.vocab)

  def one_hot_encode(self, sequence):
    tokens = self.tokenize(sequence.upper())
    one_hot = np.zeros((len(tokens), len(self.vocab)), dtype=int)
    for i, token in enumerate(tokens):
      if token in self.vocab:
        one_hot[i, self.vocab[token]] = 1
    return one_hot

  def reverse_complement(self, sequence):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', '-': '-'}
    return ''.join(complement.get(base, base) for base in reversed(sequence.upper()))

  def pad_sequence(self, sequence, target_length, pad_char='-'):
    return sequence[:target_length] if len(sequence) >= target_length else sequence + pad_char * (target_length - len(sequence))