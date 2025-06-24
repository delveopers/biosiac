import pytest
import tempfile
import os
import numpy as np
from biosaic._dna import DNA

# DNA Tests
class TestDNAInitialization:
  def test_init_continuous(self):
    dna = DNA(kmer=3, continuous=True)
    assert dna.kmer == 3
    assert dna.continuous == True
    assert dna.vocab_size == 5**3
    assert dna.has_special_tokens == False

  def test_init_non_continuous(self):
    dna = DNA(kmer=3, continuous=False)
    assert dna.kmer == 3
    assert dna.continuous == False
    expected_size = 5 + 25 + 125 + len(dna.special_tokens)
    assert dna.vocab_size == expected_size

  def test_init_with_special_tokens(self):
    special_tokens = ['<START>', '<END>']
    dna = DNA(kmer=2, continuous=False, special_tokens=special_tokens)
    assert dna.has_special_tokens == True
    assert dna.special_tokens == special_tokens
    expected_size = (5 + 25) + 2
    assert dna.vocab_size == expected_size

  def test_special_tokens_with_continuous_raises_error(self):
    with pytest.raises(ValueError):
      DNA(kmer=3, continuous=True, special_tokens=['<START>'])

class TestDNATokenization:
  def test_tokenize_continuous(self):
    dna = DNA(kmer=3, continuous=True)
    tokens = dna.tokenize("ATGCATGC")
    assert tokens == ['ATG', 'TGC', 'GCA', 'CAT', 'ATG', 'TGC']

  def test_tokenize_non_continuous(self):
    dna = DNA(kmer=3, continuous=False)
    tokens = dna.tokenize("ATGCATGC")
    assert tokens == ['ATG', 'CAT', 'GC']

  def test_tokenize_with_special_tokens(self):
    dna = DNA(kmer=2, continuous=False, special_tokens=['<S>', '</S>'])
    tokens = dna.tokenize("<S>ATGC</S>")
    assert tokens == ['<S>', 'AT', 'GC', '</S>']

  def test_tokenize_invalid_character(self):
    dna = DNA(kmer=2)
    with pytest.raises(ValueError):
      dna.tokenize("ATGCX")

class TestDNAVocabulary:
  def test_build_vocab_continuous(self):
    dna = DNA(kmer=2, continuous=True)
    dna.build_vocab()
    assert len(dna.vocab) == 25
    assert 'AA' in dna.vocab
    assert 'TT' in dna.vocab
    assert '--' in dna.vocab

  def test_build_vocab_non_continuous(self):
    dna = DNA(kmer=2, continuous=False)
    dna.build_vocab()
    assert len(dna.vocab) == 35
    assert 'A' in dna.vocab
    assert 'AA' in dna.vocab

  def test_build_vocab_with_special_tokens(self):
    dna = DNA(kmer=2, continuous=False, special_tokens=['<S>', '</S>'])
    dna.build_vocab()
    assert len(dna.vocab) == 32  # 30 + 2
    assert '<S>' in dna.vocab
    assert '</S>' in dna.vocab

class TestDNAEncodeDecode:
  def test_encode_decode_roundtrip(self):
    dna = DNA(kmer=2, continuous=True)
    dna.build_vocab()
    sequence = "ATGCATGC"
    encoded = dna.encode(sequence)
    decoded = dna.decode(encoded)
    assert decoded == sequence

  def test_encode_decode_with_special_tokens(self):
    dna = DNA(kmer=2, continuous=False, special_tokens=['<S>', '</S>'])
    dna.build_vocab()
    sequence = "<S>ATGC</S>"
    encoded = dna.encode(sequence)
    decoded = dna.decode(encoded)
    assert decoded == sequence

  def test_ids_to_chars_and_chars_to_ids(self):
    dna = DNA(kmer=2, continuous=True)
    dna.build_vocab()
    chars = ['AT', 'TG', 'GC']
    ids = dna.chars_to_ids(chars)
    converted_chars = dna.ids_to_chars(ids)
    assert chars == converted_chars

class TestDNADetokenization:
  def test_detokenize_continuous(self):
    dna = DNA(kmer=3, continuous=True)
    tokens = ['ATG', 'TGC', 'GCA']
    result = dna.detokenize(tokens)
    assert result == "ATGCA"
    
    assert dna.detokenize(['ATG']) == "ATG"
    assert dna.detokenize([]) == ""

  def test_detokenize_non_continuous(self):
    dna = DNA(kmer=2, continuous=False)
    tokens = ['AT', 'GC', 'TA']
    result = dna.detokenize(tokens)
    assert result == "ATGCTA"

class TestDNAVerification:
  def test_verify_continuous_valid(self):
    dna = DNA(kmer=3, continuous=True)
    dna.build_vocab()
    tokens = ['ATG', 'TGC', 'GCA']
    verified = dna.verify(tokens)
    assert verified[0]['match'] == True
    assert verified[1]['match'] == True

  def test_verify_continuous_invalid(self):
    dna = DNA(kmer=3, continuous=True)
    dna.build_vocab()
    tokens = ['ATG', 'CAT']
    verified = dna.verify(tokens)
    assert verified[0]['match'] == False

  def test_verify_with_special_tokens(self):
    dna = DNA(kmer=2, continuous=False, special_tokens=['<S>'])
    dna.build_vocab()
    tokens = ['<S>', 'AT', 'GC']
    verified = dna.verify(tokens)
    assert verified[0]['match'] == 'special_token'
    assert verified[1]['match'] == True

class TestDNAPersistence:
  @pytest.fixture
  def temp_dir(self):
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    for file in os.listdir(temp_dir):
      os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

  def test_save_and_load_json(self, temp_dir):
    dna = DNA(kmer=2, continuous=True)
    dna.build_vocab()
    save_path = os.path.join(temp_dir, "test_vocab")
    dna.save(save_path, as_json=True)
    
    dna2 = DNA(kmer=1)
    dna2.load(save_path + ".json")
    
    assert dna.kmer == dna2.kmer
    assert dna.continuous == dna2.continuous
    assert dna.vocab == dna2.vocab

  def test_save_and_load_pickle(self, temp_dir):
    dna = DNA(kmer=2, continuous=False, special_tokens=['<TEST>'])
    dna.build_vocab()
    save_path = os.path.join(temp_dir, "test_vocab")
    dna.save(save_path, as_json=False)
    
    dna2 = DNA(kmer=1)
    dna2.load(save_path + ".model")
    
    assert dna.kmer == dna2.kmer
    assert dna.continuous == dna2.continuous
    assert dna.special_tokens == dna2.special_tokens
    assert dna.vocab == dna2.vocab

class TestDNAUtilities:
  def test_one_hot_encode(self):
    dna = DNA(kmer=2, continuous=False)
    dna.build_vocab()
    one_hot = dna.one_hot_encode("ATGC")
    
    expected_shape = (2, len(dna.vocab))
    assert one_hot.shape == expected_shape
    for row in one_hot:
      assert np.sum(row) == 1

  def test_reverse_complement(self):
    dna = DNA(kmer=2)
    assert dna.reverse_complement("ATGC") == "GCAT"
    assert dna.reverse_complement("AT-GC") == "GC-AT"
    assert dna.reverse_complement("ATAT") == "ATAT"

  def test_pad_sequence(self):
    dna = DNA(kmer=2)
    assert dna.pad_sequence("ATG", 6) == "ATG---"
    assert dna.pad_sequence("ATGCATGC", 5) == "ATGCA"
    assert dna.pad_sequence("ATG", 3) == "ATG"
    assert dna.pad_sequence("AT", 5, pad_char='N') == "ATNNN"

class TestDNAEdgeCases:
  def test_edge_cases(self):
    dna = DNA(kmer=1, continuous=True)
    dna.build_vocab()
    tokens = dna.tokenize("ATGC")
    assert tokens == ['A', 'T', 'G', 'C']
    
    dna = DNA(kmer=2, continuous=False)
    assert dna.tokenize("") == []
    assert dna.tokenize("A") == ['A']

if __name__ == '__main__':
  pytest.main([__file__, "-v"])