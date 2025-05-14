# test_dna.py
import os, json
import pytest

from biosaic._dna import DNA

@pytest.fixture
def sample_seq():
  return "ATGC-ATGCA"

def test_tokenize_continuous(sample_seq):
  dna = DNA(kmer=3, continuous=True)
  toks = dna.tokenize(sample_seq)
  assert toks == ["ATG", "TGC", "GC-", "C-A", "-AT", "ATG", "TGC", "GCA"]

def test_tokenize_non_continuous(sample_seq):
  dna = DNA(kmer=2, continuous=False)
  toks = dna.tokenize(sample_seq)
  assert toks == ["AT", "GC", "-A", "TG", "CA"]

def test_detokenize_continuous():
  dna = DNA(kmer=3, continuous=True)
  ids = ["ATG", "TGC", "GCA"]
  assert dna.detokenize(ids) == "ATGCA"

def test_detokenize_non_continuous():
  dna = DNA(kmer=2, continuous=False)
  ids = ["AT", "GC", "TA"]
  assert dna.detokenize(ids) == "ATGCTA"

def test_build_vocab_and_encode_decode(sample_seq):
  dna = DNA(kmer=2, continuous=False)
  dna.build_vocab()
  # ensure vocab created
  assert dna.vocab_size == len(dna.vocab)
  # round-trip encode/decode
  toks = dna.tokenize("ATGC")
  ids = dna.chars_to_ids(toks)
  chars = dna.ids_to_chars(ids)
  assert chars == toks

def test_encode_invalid_characters():
  dna = DNA(kmer=2)
  with pytest.raises(ValueError):
    dna.tokenize("AXT")

def test_verify_and_save_load(tmp_path):
  dna = DNA(kmer=2, continuous=False)
  dna.build_vocab()
  toks = dna.tokenize("ATGC")
  ids = dna.chars_to_ids(toks)

  verified = dna.verify(ids, file=str(tmp_path))
  file_path = tmp_path / "verify.json"
  assert file_path.exists()
  # verify.json written
  with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    assert isinstance(data, list)
    assert "kmer1" in data[0] and "kmer2" in data[0] and "match" in data[0]

  # test save/load model
  model_path = str(tmp_path / "vocab_test")
  dna.save(model_path, as_json=False)
  loaded = DNA(kmer=1)  # dummy instance
  loaded.load(model_path + ".model")

  assert loaded.vocab == dna.vocab