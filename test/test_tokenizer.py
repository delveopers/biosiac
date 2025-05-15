# test_tokenizer.py
import pytest
import tempfile
import os

from biosaic._main import Tokenizer
from biosaic._dna import DNA
from biosaic._protein import Protein

def test_init_and_str_dna():
  tok = Tokenizer(mode="dna", kmer=2, continuous=False)
  s = str(tok)
  assert "kmer=2" in s and "continuous=False" in s

def test_init_and_str_protein():
  tok = Tokenizer(mode="protein", kmer=3, continuous=True)
  s = str(tok)
  assert "kmer=3" in s and "continuous=True" in s

def test_encode_decode_dna(monkeypatch):
  # bypass remote load by injecting a small vocab
  tok = Tokenizer(mode="dna", kmer=1)
  tok._tokenizer.vocab = {"A": 0, "T": 1}
  tok._tokenizer.ids_to_token = {0: "A", 1: "T"}
  seq = "ATTA"
  ids = tok.encode(seq)
  assert ids == [0, 1, 1, 0]
  assert tok.decode(ids) == seq

def test_tokenize_detokenize_protein():
  tok = Tokenizer(mode="protein", kmer=2, continuous=False)
  # inject custom tokenizer behavior
  tok._tokenizer.tokenize = lambda x: ["AR", "ND"]
  tok._tokenizer.detokenize = lambda x: "ARND"
  toks = tok.tokenize("ARND")
  assert toks == ["AR", "ND"]
  assert tok.detokenize(toks) == "ARND"

def test_vocab_property():
  tok = Tokenizer(mode="dna", kmer=1)
  tok._tokenizer.vocab = {"A":0}
  assert tok.vocab == {"A":0}

def test_invalid_mode():
  with pytest.raises(AssertionError):
    Tokenizer(mode="rna", kmer=2)