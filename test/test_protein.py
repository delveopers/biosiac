import pytest
import tempfile
import os
import json
import numpy as np
from biosaic._protein import Protein, AMINO_ACIDS

class TestProteinConstants:
  def test_amino_acids_constant(self):
    expected = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-']
    assert AMINO_ACIDS == expected
    assert len(AMINO_ACIDS) == 21

class TestProteinInitialization:
  def test_init_continuous(self):
    protein = Protein(kmer=3, continuous=True)
    assert protein.kmer == 3
    assert protein.continuous == True
    assert protein.vocab_size == 21**3
    assert protein.has_special_tokens == False

  def test_init_non_continuous(self):
    protein = Protein(kmer=3, continuous=False)
    assert protein.kmer == 3
    assert protein.continuous == False
    expected_size = 21 + 441 + 9261 + len(protein.special_tokens)
    assert protein.vocab_size == expected_size

  def test_init_with_special_tokens(self):
    special_tokens = ['<START>', '<END>']
    protein = Protein(kmer=2, continuous=False, special_tokens=special_tokens)
    assert protein.has_special_tokens == True
    assert protein.special_tokens == special_tokens
    expected_size = (21 + 441) + 2
    assert protein.vocab_size == expected_size

  def test_special_tokens_with_continuous_raises_error(self):
    with pytest.raises(ValueError):
      Protein(kmer=3, continuous=True, special_tokens=['<START>'])

  def test_large_kmer_vocab_size_calculation(self):
    protein = Protein(kmer=4, continuous=True)
    assert protein.vocab_size == 21**4
    
    protein_nc = Protein(kmer=4, continuous=False)
    expected_size = 21 + 21**2 + 21**3 + 21**4 + len(protein_nc.special_tokens)
    assert protein_nc.vocab_size == expected_size

class TestProteinTokenization:
  def test_tokenize_continuous(self):
    protein = Protein(kmer=3, continuous=True)
    tokens = protein.tokenize("ARNDCQ")
    assert tokens == ['ARN', 'RND', 'NDC', 'DCQ']

  def test_tokenize_non_continuous(self):
    protein = Protein(kmer=3, continuous=False)
    tokens = protein.tokenize("ARNDCQE")
    assert tokens == ['ARN', 'DCQ', 'E']

  def test_tokenize_with_special_tokens(self):
    protein = Protein(kmer=2, continuous=False, special_tokens=['<S>', '</S>'])
    tokens = protein.tokenize("<S>ARND</S>")
    assert tokens == ['<S>', 'AR', 'ND', '</S>']

  def test_tokenize_mixed_special_tokens(self):
    protein = Protein(kmer=2, continuous=False, special_tokens=['<P>', '<M>'])
    tokens = protein.tokenize("AR<P>ND<M>CQ")
    assert tokens == ['AR', '<P>', 'ND', '<M>', 'CQ']

  def test_tokenize_invalid_character(self):
    protein = Protein(kmer=2)
    with pytest.raises(ValueError):
      protein.tokenize("ARNDX")

  def test_all_amino_acids_tokenization(self):
    protein = Protein(kmer=1, continuous=False)
    protein.build_vocab()
    sequence = "ARNDCQEGHILKMFPSTWYV-"
    tokens = protein.tokenize(sequence)
    assert len(tokens) == 21
    assert tokens == list(sequence)

  def test_case_insensitive_processing(self):
    protein = Protein(kmer=2, continuous=False)
    protein.build_vocab()
    encoded_lower = protein.encode("arnd")
    encoded_upper = protein.encode("ARND")
    assert encoded_lower == encoded_upper

class TestProteinVocabulary:
  def test_build_vocab_continuous(self):
    protein = Protein(kmer=2, continuous=True)
    protein.build_vocab()
    assert len(protein.vocab) == 441
    assert 'AA' in protein.vocab
    assert 'RR' in protein.vocab
    assert '--' in protein.vocab

  def test_build_vocab_non_continuous(self):
    protein = Protein(kmer=2, continuous=False)
    protein.build_vocab()
    assert len(protein.vocab) == 467
    assert 'A' in protein.vocab
    assert 'AA' in protein.vocab

  def test_build_vocab_with_special_tokens(self):
    protein = Protein(kmer=2, continuous=False, special_tokens=['<S>', '</S>'])
    protein.build_vocab()
    assert len(protein.vocab) == 464  # 462 + 2
    assert '<S>' in protein.vocab
    assert '</S>' in protein.vocab

class TestProteinEncodeDecode:
  def test_encode_decode_roundtrip(self):
    protein = Protein(kmer=2, continuous=True)
    protein.build_vocab()
    sequence = "ARNDCQEGHIL"
    encoded = protein.encode(sequence)
    decoded = protein.decode(encoded)
    assert decoded == sequence

  def test_encode_decode_with_special_tokens(self):
    protein = Protein(kmer=2, continuous=False, special_tokens=['<S>', '</S>'])
    protein.build_vocab()
    sequence = "<S>ARND</S>"
    encoded = protein.encode(sequence)
    decoded = protein.decode(encoded)
    assert decoded == sequence

  def test_ids_to_chars_and_chars_to_ids(self):
    protein = Protein(kmer=2, continuous=True)
    protein.build_vocab()
    chars = ['AR', 'RN', 'ND']
    ids = protein.chars_to_ids(chars)
    converted_chars = protein.ids_to_chars(ids)
    assert chars == converted_chars

class TestProteinDetokenization:
  def test_detokenize_continuous(self):
    protein = Protein(kmer=3, continuous=True)
    tokens = ['ARN', 'RND', 'NDC']
    result = protein.detokenize(tokens)
    assert result == "ARNDC"
    
    assert protein.detokenize(['ARN']) == "ARN"
    assert protein.detokenize([]) == ""

  def test_detokenize_non_continuous(self):
    protein = Protein(kmer=2, continuous=False)
    tokens = ['AR', 'ND', 'CQ']
    result = protein.detokenize(tokens)
    assert result == "ARNDCQ"

class TestProteinVerification:
  def test_verify_continuous_valid(self):
    protein = Protein(kmer=3, continuous=True)
    protein.build_vocab()
    tokens = ['ARN', 'RND', 'NDC']
    verified = protein.verify(tokens)
    assert verified[0]['match'] == True
    assert verified[1]['match'] == True

  def test_verify_continuous_invalid(self):
    protein = Protein(kmer=3, continuous=True)
    protein.build_vocab()
    tokens = ['ARN', 'CQE']
    verified = protein.verify(tokens)
    assert verified[0]['match'] == False

  def test_verify_with_special_tokens(self):
    protein = Protein(kmer=2, continuous=False, special_tokens=['<S>'])
    protein.build_vocab()
    tokens = ['<S>', 'AR', 'ND']
    verified = protein.verify(tokens)
    assert verified[0]['match'] == 'special_token'
    assert verified[1]['match'] == True

class TestProteinPersistence:
  @pytest.fixture
  def temp_dir(self):
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    for file in os.listdir(temp_dir):
      os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

  def test_save_and_load_json(self, temp_dir):
    protein = Protein(kmer=2, continuous=True)
    protein.build_vocab()
    save_path = os.path.join(temp_dir, "test_protein_vocab")
    protein.save(save_path, as_json=True)
    
    protein2 = Protein(kmer=2, continuous=True)
    protein2.load(save_path + ".json")
    
    assert protein.kmer == protein2.kmer
    assert protein.continuous == protein2.continuous
    assert protein.vocab == protein2.vocab

  def test_save_and_load_pickle(self, temp_dir):
    protein = Protein(kmer=2, continuous=False, special_tokens=['<TEST>'])
    protein.build_vocab()
    save_path = os.path.join(temp_dir, "test_protein_vocab")
    protein.save(save_path, as_json=False)
    
    protein2 = Protein(kmer=2)
    protein2.load(save_path + ".model")
    
    assert protein.kmer == protein2.kmer
    assert protein.continuous == protein2.continuous
    assert protein.special_tokens == protein2.special_tokens
    assert protein.vocab == protein2.vocab

  def test_verify_save_to_file(self, temp_dir):
    protein = Protein(kmer=2, continuous=True)
    protein.build_vocab()
    tokens = ['AR', 'RN']
    verified = protein.verify(tokens, file=temp_dir)
    
    verify_file = os.path.join(temp_dir, "verify.json")
    assert os.path.exists(verify_file)
    
    with open(verify_file, 'r') as f:
      saved_data = json.load(f)
    assert len(saved_data) == 1
    assert saved_data[0]['kmer1'] == 'AR'
    assert saved_data[0]['kmer2'] == 'RN'

class TestProteinUtilities:
  def test_one_hot_encode(self):
    protein = Protein(kmer=2, continuous=False)
    protein.build_vocab()
    one_hot = protein.one_hot_encode("ARND")
    
    expected_shape = (2, len(protein.vocab))
    assert one_hot.shape == expected_shape
    for row in one_hot:
      assert np.sum(row) == 1

  def test_pad_sequence(self):
    protein = Protein(kmer=2)
    assert protein.pad_sequence("ARN", 6) == "ARN---"
    assert protein.pad_sequence("ARNDCQEGHIL", 5) == "ARNDC"
    assert protein.pad_sequence("ARN", 3) == "ARN"
    assert protein.pad_sequence("AR", 5, pad_char='X') == "ARXXX"

  def test_reverse_complement_not_implemented(self):
    protein = Protein(kmer=2)
    with pytest.raises(NotImplementedError, match="Proteins don't have reverse complement"):
      protein.reverse_complement("ARND")

class TestProteinEdgeCases:
  def test_edge_cases(self):
    protein = Protein(kmer=1, continuous=True)
    protein.build_vocab()
    tokens = protein.tokenize("ARND")
    assert tokens == ['A', 'R', 'N', 'D']
    
    protein = Protein(kmer=2, continuous=False)
    assert protein.tokenize("") == []
    assert protein.tokenize("A") == ['A']

if __name__ == '__main__':
  pytest.main([__file__, "-v"])