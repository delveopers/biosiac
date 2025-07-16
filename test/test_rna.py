import pytest
import tempfile
import os
import numpy as np
from biosaic._rna import RNA

# DNA Tests
class TestDNAInitialization:
  def test_init_continuous(self):
    rna = RNA(kmer=3, continuous=True)
    assert rna.kmer == 3
    assert rna.continuous == True
    assert rna.vocab_size == 5**3
    assert rna.has_special_tokens == False

  def test_init_non_continuous(self):
    rna = RNA(kmer=3, continuous=False)
    assert rna.kmer == 3
    assert rna.continuous == False
    expected_size = 5 + 25 + 125 + len(rna.special_tokens)
    assert rna.vocab_size == expected_size

  def test_init_with_special_tokens(self):
    special_tokens = ['<START>', '<END>']
    rna = RNA(kmer=2, continuous=False, special_tokens=special_tokens)
    assert rna.has_special_tokens == True
    assert rna.special_tokens == special_tokens
    expected_size = (5 + 25) + 2
    assert rna.vocab_size == expected_size

  def test_special_tokens_with_continuous_raises_error(self):
    with pytest.raises(ValueError):
      RNA(kmer=3, continuous=True, special_tokens=['<START>'])

class TestDNATokenization:
  def test_tokenize_continuous(self):
    rna = RNA(kmer=3, continuous=True)
    tokens = rna.tokenize("AUGCAUGC")
    assert tokens == ['AUG', 'UGC', 'GCA', 'CAU', 'AUG', 'UGC']

  def test_tokenize_non_continuous(self):
    rna = RNA(kmer=3, continuous=False)
    tokens = rna.tokenize("AUGCAUGC")
    assert tokens == ['AUG', 'CAU', 'GC']

  def test_tokenize_with_special_tokens(self):
    rna = RNA(kmer=2, continuous=False, special_tokens=['<S>', '</S>'])
    tokens = rna.tokenize("<S>AUGC</S>")
    assert tokens == ['<S>', 'AU', 'GC', '</S>']

  def test_tokenize_invalid_character(self):
    rna = RNA(kmer=2)
    with pytest.raises(ValueError):
      rna.tokenize("AUGCX")

class TestDNAVocabulary:
  def test_build_vocab_continuous(self):
    rna = RNA(kmer=2, continuous=True)
    rna.build_vocab()
    assert len(rna.vocab) == 25
    assert 'AA' in rna.vocab
    assert 'UU' in rna.vocab
    assert '--' in rna.vocab

  def test_build_vocab_non_continuous(self):
    rna = RNA(kmer=2, continuous=False)
    rna.build_vocab()
    assert len(rna.vocab) == 35
    assert 'A' in rna.vocab
    assert 'AA' in rna.vocab

  def test_build_vocab_with_special_tokens(self):
    rna = RNA(kmer=2, continuous=False, special_tokens=['<S>', '</S>'])
    rna.build_vocab()
    assert len(rna.vocab) == 32  # 30 + 2
    assert '<S>' in rna.vocab
    assert '</S>' in rna.vocab

class TestDNAEncodeDecode:
  def test_encode_decode_roundtrip(self):
    rna = RNA(kmer=2, continuous=True)
    rna.build_vocab()
    sequence = "AUGCAUGC"
    encoded = rna.encode(sequence)
    decoded = rna.decode(encoded)
    assert decoded == sequence

  def test_encode_decode_with_special_tokens(self):
    rna = RNA(kmer=2, continuous=False, special_tokens=['<S>', '</S>'])
    rna.build_vocab()
    sequence = "<S>AUGC</S>"
    encoded = rna.encode(sequence)
    decoded = rna.decode(encoded)
    assert decoded == sequence

  def test_ids_to_chars_and_chars_to_ids(self):
    rna = RNA(kmer=2, continuous=True)
    rna.build_vocab()
    chars = ['AU', 'UG', 'GC']
    ids = rna.chars_to_ids(chars)
    converted_chars = rna.ids_to_chars(ids)
    assert chars == converted_chars

class TestDNADetokenization:
  def test_detokenize_continuous(self):
    rna = RNA(kmer=3, continuous=True)
    tokens = ['AUG', 'UGC', 'GCA']
    result = rna.detokenize(tokens)
    assert result == "AUGCA"
    
    assert rna.detokenize(['AUG']) == "AUG"
    assert rna.detokenize([]) == ""

  def test_detokenize_non_continuous(self):
    rna = RNA(kmer=2, continuous=False)
    tokens = ['AU', 'GC', 'UA']
    result = rna.detokenize(tokens)
    assert result == "AUGCUA"

class TestDNAVerification:
  def test_verify_continuous_valid(self):
    rna = RNA(kmer=3, continuous=True)
    rna.build_vocab()
    tokens = ['AUG', 'UGC', 'GCA']
    verified = rna.verify(tokens)
    assert verified[0]['match'] == True
    assert verified[1]['match'] == True

  def test_verify_continuous_invalid(self):
    rna = RNA(kmer=3, continuous=True)
    rna.build_vocab()
    tokens = ['AUG', 'CAU']
    verified = rna.verify(tokens)
    assert verified[0]['match'] == False

  def test_verify_with_special_tokens(self):
    rna = RNA(kmer=2, continuous=False, special_tokens=['<S>'])
    rna.build_vocab()
    tokens = ['<S>', 'AU', 'GC']
    verified = rna.verify(tokens)
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
    rna = RNA(kmer=2, continuous=True)
    rna.build_vocab()
    save_path = os.path.join(temp_dir, "test_vocab")
    rna.save(save_path, as_json=True)
    
    dna2 = RNA(kmer=1)
    dna2.load(save_path + ".json")
    
    assert rna.kmer == dna2.kmer
    assert rna.continuous == dna2.continuous
    assert rna.vocab == dna2.vocab

  def test_save_and_load_pickle(self, temp_dir):
    rna = RNA(kmer=2, continuous=False, special_tokens=['<TEST>'])
    rna.build_vocab()
    save_path = os.path.join(temp_dir, "test_vocab")
    rna.save(save_path, as_json=False)
    
    dna2 = RNA(kmer=1)
    dna2.load(save_path + ".model")
    
    assert rna.kmer == dna2.kmer
    assert rna.continuous == dna2.continuous
    assert rna.special_tokens == dna2.special_tokens
    assert rna.vocab == dna2.vocab

class TestDNAUtilities:
  def test_one_hot_encode(self):
    rna = RNA(kmer=2, continuous=False)
    rna.build_vocab()
    one_hot = rna.one_hot_encode("AUGC")
    
    expected_shape = (2, len(rna.vocab))
    assert one_hot.shape == expected_shape
    for row in one_hot:
      assert np.sum(row) == 1

  def test_reverse_complement(self):
    rna = RNA(kmer=2)
    assert rna.reverse_complement("AUGC") == "GCAU"
    assert rna.reverse_complement("AU-GC") == "GC-AU"
    assert rna.reverse_complement("AUAU") == "AUAU"

  def test_pad_sequence(self):
    rna = RNA(kmer=2)
    assert rna.pad_sequence("AUG", 6) == "AUG---"
    assert rna.pad_sequence("AUGCAUGC", 5) == "AUGCA"
    assert rna.pad_sequence("AUG", 3) == "AUG"
    assert rna.pad_sequence("AU", 5, pad_char='N') == "AUNNN"

class TestDNAEdgeCases:
  def test_edge_cases(self):
    rna = RNA(kmer=1, continuous=True)
    rna.build_vocab()
    tokens = rna.tokenize("AUGC")
    assert tokens == ['A', 'U', 'G', 'C']
    
    rna = RNA(kmer=2, continuous=False)
    assert rna.tokenize("") == []
    assert rna.tokenize("A") == ['A']

if __name__ == '__main__':
  pytest.main([__file__, "-v"])