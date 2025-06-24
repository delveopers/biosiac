import unittest
import numpy as np
from biosaic._protein import Protein, AMINO_ACIDS

class TestProteinTokenizer(unittest.TestCase):
  def setUp(self):
    self.protein_continuous = Protein(kmer=3, continuous=True)
    self.protein_non_continuous = Protein(kmer=3, continuous=False)
    self.protein_with_special = Protein(kmer=2, continuous=False, special_tokens=['<START>', '<END>'])
    
  def test_initialization(self):
    # Test basic initialization
    self.assertEqual(self.protein_continuous.kmer, 3)
    self.assertTrue(self.protein_continuous.continuous)
    self.assertEqual(len(self.protein_continuous._base_chars), 21)  # 20 AA + gap
    
    # Test vocab size calculation
    self.assertEqual(self.protein_continuous.vocab_size, 21**3)  # 9261
    expected_non_cont = sum(21**i for i in range(1, 4))
    self.assertEqual(self.protein_non_continuous.vocab_size, expected_non_cont)
    
  def test_tokenization_continuous(self):
    seq = "ACDEFGH"
    tokens = self.protein_continuous.tokenize(seq)
    expected = ["ACD", "CDE", "DEF", "EFG", "FGH"]
    self.assertEqual(tokens, expected)
    
  def test_tokenization_non_continuous(self):
    seq = "ACDEFGH"
    tokens = self.protein_non_continuous.tokenize(seq)
    expected = ["ACD", "EFG", "H"]  # Non-overlapping 3-mers
    self.assertEqual(tokens, expected)
    
  def test_tokenization_with_special_tokens(self):
    seq = "<START>ACDE<END>FGHI"
    tokens = self.protein_with_special.tokenize(seq)
    expected = ["<START>", "AC", "DE", "<END>", "FG", "HI"]
    self.assertEqual(tokens, expected)
    
  def test_invalid_characters(self):
    with self.assertRaises(ValueError):
      self.protein_continuous.tokenize("ACDEX")  # X not in standard amino acids
      
  def test_build_vocab_and_encode_decode(self):
    self.protein_continuous.build_vocab()
    seq = "ACDEFG"
    
    # Test encoding
    encoded = self.protein_continuous.encode(seq)
    self.assertIsInstance(encoded, list)
    self.assertTrue(all(isinstance(x, int) for x in encoded))
    
    # Test decoding
    decoded = self.protein_continuous.decode(encoded)
    self.assertEqual(decoded, seq)
    
  def test_verify_continuous(self):
    self.protein_continuous.build_vocab()
    tokens = ["ACD", "CDE", "DEF"]
    verified = self.protein_continuous.verify(tokens)
    
    self.assertEqual(len(verified), 2)
    self.assertTrue(verified[0]["match"])  # ACD->CDE: CD matches
    self.assertTrue(verified[1]["match"])  # CDE->DEF: DE matches
    
  def test_one_hot_encoding(self):
    self.protein_continuous.build_vocab()
    seq = "ACDE"
    one_hot = self.protein_continuous.one_hot_encode(seq)
    
    expected_tokens = len(self.protein_continuous.tokenize(seq))
    self.assertEqual(one_hot.shape, (expected_tokens, len(self.protein_continuous.vocab)))
    
  def test_reverse_complement_error(self):
    # Should raise NotImplementedError with humorous message
    with self.assertRaises(NotImplementedError) as context:
      self.protein_continuous.reverse_complement("ACDE")
    self.assertIn("dumbass", str(context.exception).lower())
    
  def test_pad_sequence(self):
    seq = "ACDE"
    padded = self.protein_continuous.pad_sequence(seq, 8)
    self.assertEqual(padded, "ACDE----")
    
  def test_detokenization_continuous_vs_non_continuous(self):
    # Test continuous detokenization
    tokens_cont = ["ACD", "CDE", "DEF"]
    result_cont = self.protein_continuous.detokenize(tokens_cont)
    self.assertEqual(result_cont, "ACDEF")  # Overlapping chars removed
    
    # Test non-continuous detokenization
    tokens_non_cont = ["ACD", "EFG", "HI"]
    result_non_cont = self.protein_non_continuous.detokenize(tokens_non_cont)
    self.assertEqual(result_non_cont, "ACDEFGHI")  # Simple concatenation

class TestEdgeCases(unittest.TestCase):
  def test_special_tokens_only_sequence(self):
    protein = Protein(kmer=2, continuous=False, special_tokens=['<S>', '<E>'])
    protein.build_vocab()
    
    seq = "<S><E>"
    tokens = protein.tokenize(seq)
    self.assertEqual(tokens, ["<S>", "<E>"])
    
    encoded = protein.encode(seq)
    decoded = protein.decode(encoded)
    self.assertEqual(decoded, seq)

if __name__ == "__main__":
  # Run all tests
  unittest.main(verbosity=2)