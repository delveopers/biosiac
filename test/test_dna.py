import unittest
import tempfile
import os
from biosaic._dna import DNA

class TestDNATokenizer(unittest.TestCase):
  def setUp(self):
    self.dna_continuous = DNA(kmer=3, continuous=True)
    self.dna_non_continuous = DNA(kmer=3, continuous=False)
    self.dna_with_special = DNA(kmer=2, continuous=False, special_tokens=['<START>', '<END>'])

  def test_initialization(self):
    # Test basic initialization
    self.assertEqual(self.dna_continuous.kmer, 3)
    self.assertTrue(self.dna_continuous.continuous)
    self.assertEqual(self.dna_continuous._base_chars, ['A', 'T', 'G', 'C', '-'])
    
    # Test vocab size calculation
    self.assertEqual(self.dna_continuous.vocab_size, 5**3)  # 125
    expected_non_cont = sum(5**i for i in range(1, 4))  # 5 + 25 + 125 = 155
    self.assertEqual(self.dna_non_continuous.vocab_size, expected_non_cont)
    
    # Test special tokens error with continuous=True
    with self.assertRaises(ValueError):
      DNA(kmer=2, continuous=True, special_tokens=['<S>'])

  def test_tokenization_continuous(self):
    seq = "ATGCATGC"
    tokens = self.dna_continuous.tokenize(seq)
    expected = ["ATG", "TGC", "GCA", "CAT", "ATG", "TGC"]
    self.assertEqual(tokens, expected)
    
    # Edge case: short sequence
    short_seq = "AT"
    short_tokens = self.dna_continuous.tokenize(short_seq)
    self.assertEqual(short_tokens, [])
    
  def test_tokenization_non_continuous(self):
    seq = "ATGCATGC"
    tokens = self.dna_non_continuous.tokenize(seq)
    expected = ["ATG", "CAT", "GC"]  # Non-overlapping 3-mers
    self.assertEqual(tokens, expected)
    
  def test_tokenization_with_special_tokens(self):
    seq = "<START>ATGC<END>GGCC"
    tokens = self.dna_with_special.tokenize(seq)
    expected = ["<START>", "AT", "GC", "<END>", "GG", "CC"]
    self.assertEqual(tokens, expected)
    
  def test_invalid_characters(self):
    with self.assertRaises(ValueError):
      self.dna_continuous.tokenize("ATGCX")  # X is invalid
      
  def test_build_vocab_and_encode_decode(self):
    self.dna_continuous.build_vocab()
    seq = "ATGCAT"
    
    # Test encoding
    encoded = self.dna_continuous.encode(seq)
    self.assertIsInstance(encoded, list)
    self.assertTrue(all(isinstance(x, int) for x in encoded))
    
    # Test decoding
    decoded = self.dna_continuous.decode(encoded)
    self.assertEqual(decoded, seq)
    
  def test_ids_to_chars_and_chars_to_ids(self):
    self.dna_continuous.build_vocab()
    tokens = ["ATG", "TGC", "GCA"]
    
    # Test chars to ids
    ids = self.dna_continuous.chars_to_ids(tokens)
    self.assertTrue(all(isinstance(x, int) for x in ids))
    
    # Test ids to chars
    chars = self.dna_continuous.ids_to_chars(ids)
    self.assertEqual(chars, tokens)
    
  def test_verify_continuous(self):
    self.dna_continuous.build_vocab()
    tokens = ["ATG", "TGC", "GCA", "CAT"]
    verified = self.dna_continuous.verify(tokens)
    
    self.assertEqual(len(verified), 3)
    self.assertTrue(verified[0]["match"])  # ATG->TGC: G matches
    self.assertTrue(verified[1]["match"])  # TGC->GCA: GC matches
    self.assertTrue(verified[2]["match"])  # GCA->CAT: CA matches
    
  def test_verify_with_special_tokens(self):
    self.dna_with_special.build_vocab()
    tokens = ["<START>", "AT", "<END>", "GC"]
    verified = self.dna_with_special.verify(tokens)
    
    self.assertEqual(len(verified), 3)
    self.assertEqual(verified[0]["match"], "special_token")
    self.assertEqual(verified[2]["match"], "special_token")
    
  def test_one_hot_encoding(self):
    self.dna_continuous.build_vocab()
    seq = "ATGC"
    one_hot = self.dna_continuous.one_hot_encode(seq)
    
    # Should have shape (num_tokens, vocab_size)
    expected_tokens = len(self.dna_continuous.tokenize(seq))
    self.assertEqual(one_hot.shape, (expected_tokens, len(self.dna_continuous.vocab)))
    self.assertEqual(one_hot.dtype, int)
    
  def test_reverse_complement(self):
    seq = "ATGC"
    rev_comp = self.dna_continuous.reverse_complement(seq)
    self.assertEqual(rev_comp, "GCAT")
    
    # Test with gaps
    seq_with_gap = "ATG-C"
    rev_comp_gap = self.dna_continuous.reverse_complement(seq_with_gap)
    self.assertEqual(rev_comp_gap, "G-CAT")
    
  def test_pad_sequence(self):
    seq = "ATGC"
    padded = self.dna_continuous.pad_sequence(seq, 8)
    self.assertEqual(padded, "ATGC----")
    
    # Test truncation
    truncated = self.dna_continuous.pad_sequence(seq, 2)
    self.assertEqual(truncated, "AT")
    
  def test_save_and_load(self):
    self.dna_continuous.build_vocab()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
      model_path = os.path.join(tmp_dir, "test_model")
      
      # Test saving as pickle
      self.dna_continuous.save(model_path)
      
      # Test loading
      new_dna = DNA(kmer=1)  # Different initial params
      new_dna.load(model_path + ".model")
      
      self.assertEqual(new_dna.kmer, 3)
      self.assertEqual(new_dna.vocab_size, self.dna_continuous.vocab_size)
      self.assertEqual(new_dna.vocab, self.dna_continuous.vocab)
      
      # Test saving as JSON
      self.dna_continuous.save(model_path, as_json=True)
      json_dna = DNA(kmer=1)
      json_dna.load(model_path + ".json")
      self.assertEqual(json_dna.vocab, self.dna_continuous.vocab)

class TestEdgeCases(unittest.TestCase):
  def test_empty_sequences(self):
    dna = DNA(kmer=3, continuous=True)
    dna.build_vocab()
    
    # Empty sequence should return empty list
    self.assertEqual(dna.tokenize(""), [])
    self.assertEqual(dna.encode(""), [])
    
  def test_sequence_shorter_than_kmer(self):
    dna = DNA(kmer=5, continuous=True)
    short_seq = "ATG"  # Length 3, kmer=5
    
    tokens = dna.tokenize(short_seq)
    self.assertEqual(tokens, [])  # No valid k-mers
    
  def test_case_insensitive(self):
    dna = DNA(kmer=2, continuous=True)
    dna.build_vocab()
    
    lower_seq = "atgc"
    upper_seq = "ATGC"
    
    # Should produce same encoding regardless of case
    self.assertEqual(dna.encode(lower_seq), dna.encode(upper_seq))

if __name__ == "__main__":
  # Run all tests
  unittest.main(verbosity=2)