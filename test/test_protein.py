# test_protein.py
import os
import json
import pytest

from biosaic._protein import Protein, AMINO_ACIDS

@pytest.fixture
def sample_seq():
  return "ARND-CEQ"

def test_tokenize_continuous(sample_seq):
  prot = Protein(kmer=2, continuous=True)
  toks = prot.tokenize(sample_seq)
  assert toks == ["AR", "RN", "ND", "D-", "-C", "CE", "EQ"]

def test_tokenize_non_continuous(sample_seq):
  prot = Protein(kmer=3, continuous=False)
  toks = prot.tokenize(sample_seq)
  assert toks == ["ARN", "D-C", "EQ"]

def test_detokenize_continuous():
  prot = Protein(kmer=2, continuous=True)
  toks = ["AR", "RN", "ND"]
  assert prot.detokenize(toks) == "ARND"

def test_detokenize_non_continuous():
  prot = Protein(kmer=2, continuous=False)
  toks = ["AR", "ND", "CE"]
  assert prot.detokenize(toks) == "ARNDCE"

def test_build_vocab_and_encode_decode(sample_seq):
  prot = Protein(kmer=1, continuous=False)
  prot.build_vocab()
  assert prot.vocab_size == len(prot.vocab)
  toks = prot.tokenize("ARND")
  ids = prot.chars_to_ids(toks)
  chars = prot.ids_to_chars(ids)
  assert chars == toks

def test_encode_invalid_characters():
  prot = Protein(kmer=2)
  with pytest.raises(ValueError):
    prot.tokenize("ARZ")

def test_verify_and_save_load(tmp_path):
  prot = Protein(kmer=2, continuous=False)
  prot.build_vocab()
  toks = prot.tokenize("ARND")
  ids = prot.chars_to_ids(toks)
  result = prot.verify(ids, file=str(tmp_path))
  file_path = tmp_path / "verify.json"
  assert file_path.exists()
  with open(file_path, "r", encoding="utf-8") as fp:
    data = json.load(fp)
    assert isinstance(data, list)
    assert "kmer1" in data[0] and "kmer2" in data[0] and "match" in data[0]

  model_path = str(tmp_path / "prot_vocab")
  prot.save(model_path, as_json=True)

  loaded = Protein(kmer=1)
  loaded.load(model_path + ".json")
  assert loaded.vocab == prot.vocab