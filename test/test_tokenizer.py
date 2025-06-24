import pytest
from unittest.mock import Mock, patch
from biosaic import Tokenizer

class TestTokenizerInitialization:
  def test_dna_tokenizer_init(self):
    with patch('biosaic._main.DNA') as mock_dna:
      mock_instance = Mock()
      mock_instance.has_special_tokens = False
      mock_dna.return_value = mock_instance
      
      tokenizer = Tokenizer("dna", 3, continuous=False)
      assert tokenizer.kmer == 3
      assert tokenizer.continuous == False
      assert tokenizer.encoding == "dna/base_3k"
      mock_dna.assert_called_once_with(kmer=3, continuous=False, special_tokens=None)

  def test_protein_tokenizer_init(self):
    with patch('biosaic._main.Protein') as mock_protein:
      mock_instance = Mock()
      mock_instance.has_special_tokens = True
      mock_protein.return_value = mock_instance
      
      tokenizer = Tokenizer("protein", 2, continuous=True, special_tokens=['<s>', '</s>'])
      assert tokenizer.kmer == 2
      assert tokenizer.continuous == True
      assert tokenizer.encoding == "protein/special_2k"
      mock_protein.assert_called_once_with(kmer=2, continuous=True, special_tokens=['<s>', '</s>'])

  def test_invalid_mode_raises_error(self):
    with pytest.raises(AssertionError, match="Unknown mode type"):
      Tokenizer("rna", 3)

  def test_dna_kmer_limit(self):
    with pytest.raises(AssertionError, match="KMer size supported only till 8"):
      Tokenizer("dna", 9)

  def test_protein_kmer_limit(self):
    with pytest.raises(AssertionError, match="KMer size supported only till 4"):
      Tokenizer("protein", 5)

  def test_encoding_path_construction(self):
    with patch('biosaic._main.DNA') as mock_dna:
      mock_instance = Mock()
      mock_instance.has_special_tokens = False
      mock_dna.return_value = mock_instance
      
      tokenizer = Tokenizer("dna", 4, continuous=True)
      expected_path = "https://raw.githubusercontent.com/delveopers/biosaic/main/vocab/dna/cont_4k.model"
      assert tokenizer.encoding_path == expected_path

class TestTokenizerMethods:
  @pytest.fixture
  def mock_tokenizer(self):
    with patch('biosaic._main.DNA') as mock_dna_class:
      mock_dna = Mock()
      mock_dna.has_special_tokens = False
      mock_dna.vocab_size = 64
      mock_dna.vocab = {'ATG': 0, 'TGC': 1, 'GCA': 2}
      mock_dna_class.return_value = mock_dna
      
      tokenizer = Tokenizer("dna", 3)
      return tokenizer, mock_dna

  def test_encode(self, mock_tokenizer):
    tokenizer, mock_dna = mock_tokenizer
    mock_dna.encode.return_value = [0, 1, 2]
    
    result = tokenizer.encode("ATGTGCGCA")
    assert result == [0, 1, 2]
    mock_dna.encode.assert_called_once_with("ATGTGCGCA")

  def test_decode(self, mock_tokenizer):
    tokenizer, mock_dna = mock_tokenizer
    mock_dna.decode.return_value = "ATGTGCGCA"
    
    result = tokenizer.decode([0, 1, 2])
    assert result == "ATGTGCGCA"
    mock_dna.decode.assert_called_once_with([0, 1, 2])

  def test_tokenize(self, mock_tokenizer):
    tokenizer, mock_dna = mock_tokenizer
    mock_dna.tokenize.return_value = ["ATG", "TGC", "GCA"]
    
    result = tokenizer.tokenize("ATGTGCGCA")
    assert result == ["ATG", "TGC", "GCA"]
    mock_dna.tokenize.assert_called_once_with("ATGTGCGCA")

  def test_detokenize(self, mock_tokenizer):
    tokenizer, mock_dna = mock_tokenizer
    mock_dna.detokenize.return_value = "ATGTGCGCA"
    
    result = tokenizer.detokenize(["ATG", "TGC", "GCA"])
    assert result == "ATGTGCGCA"
    mock_dna.detokenize.assert_called_once_with(["ATG", "TGC", "GCA"])

  def test_one_hot(self, mock_tokenizer):
    tokenizer, mock_dna = mock_tokenizer
    mock_dna.one_hot_encode.return_value = [[1, 0, 0, 0], [0, 1, 0, 0]]
    
    result = tokenizer.one_hot("AT")
    assert result == [[1, 0, 0, 0], [0, 1, 0, 0]]
    mock_dna.one_hot_encode.assert_called_once_with("AT")

  def test_reverse_complement(self, mock_tokenizer):
    tokenizer, mock_dna = mock_tokenizer
    mock_dna.reverse_complement.return_value = "CGCAT"
    
    result = tokenizer.reverse_complement("ATGCG")
    assert result == "CGCAT"
    mock_dna.reverse_complement.assert_called_once_with("ATGCG")

  def test_pad_sequence(self, mock_tokenizer):
    tokenizer, mock_dna = mock_tokenizer
    mock_dna.pad_sequence.return_value = "ATGC--"
    
    result = tokenizer.pad_sequence("ATGC", 6, "-")
    assert result == "ATGC--"
    mock_dna.pad_sequence.assert_called_once_with("ATGC", 6, "-")

  def test_vocab_size_property(self, mock_tokenizer):
    tokenizer, mock_dna = mock_tokenizer
    assert tokenizer.vocab_size == 64

  def test_vocab_property(self, mock_tokenizer):
    tokenizer, mock_dna = mock_tokenizer
    expected_vocab = {'ATG': 0, 'TGC': 1, 'GCA': 2}
    assert tokenizer.vocab == expected_vocab

class TestSpecialTokens:
  def test_default_special_tokens(self):
    with patch('biosaic._main.DNA') as mock_dna_class:
      mock_dna = Mock()
      mock_dna.has_special_tokens = True
      mock_dna_class.return_value = mock_dna
      
      tokenizer = Tokenizer("dna", 3, special_tokens=None)
      mock_dna_class.assert_called_once_with(kmer=3, continuous=False, special_tokens=None)

  def test_custom_special_tokens(self):
    with patch('biosaic._main.Protein') as mock_protein_class:
      mock_protein = Mock()
      mock_protein.has_special_tokens = True
      mock_protein_class.return_value = mock_protein
      
      custom_tokens = ['<start>', '<end>', '<pad>']
      tokenizer = Tokenizer("protein", 2, special_tokens=custom_tokens)
      mock_protein_class.assert_called_once_with(kmer=2, continuous=False, special_tokens=custom_tokens)

  def test_no_special_tokens(self):
    with patch('biosaic._main.DNA') as mock_dna_class:
      mock_dna = Mock()
      mock_dna.has_special_tokens = False
      mock_dna_class.return_value = mock_dna
      
      tokenizer = Tokenizer("dna", 3, special_tokens=False)
      assert tokenizer.encoding == "dna/base_3k"
      mock_dna_class.assert_called_once_with(kmer=3, continuous=False, special_tokens=False)

class TestEncodingModes:
  def test_continuous_encoding(self):
    with patch('biosaic._main.DNA') as mock_dna_class:
      mock_dna = Mock()
      mock_dna.has_special_tokens = False
      mock_dna_class.return_value = mock_dna
      
      tokenizer = Tokenizer("dna", 4, continuous=True)
      assert tokenizer.encoding == "dna/cont_4k"

  def test_base_encoding(self):
    with patch('biosaic._main.Protein') as mock_protein_class:
      mock_protein = Mock()
      mock_protein.has_special_tokens = False
      mock_protein_class.return_value = mock_protein
      
      tokenizer = Tokenizer("protein", 3, continuous=False)
      assert tokenizer.encoding == "protein/base_3k"

  def test_special_encoding_override(self):
    with patch('biosaic._main.DNA') as mock_dna_class:
      mock_dna = Mock()
      mock_dna.has_special_tokens = True
      mock_dna_class.return_value = mock_dna
      
      tokenizer = Tokenizer("dna", 3, continuous=True)
      assert tokenizer.encoding == "dna/special_3k"

class TestStringRepresentation:
  def test_str_with_special_tokens(self):
    with patch('biosaic._main.Protein') as mock_protein_class:
      mock_protein = Mock()
      mock_protein.has_special_tokens = True
      mock_protein_class.return_value = mock_protein
      
      tokenizer = Tokenizer("protein", 2, continuous=True, special_tokens=['<s>', '</s>'])
      result = str(tokenizer)
      assert "kmer=2" in result
      assert "continuous=True" in result
      assert "special_tokens=2" in result

class TestIntegrationScenarios:
  @patch('biosaic._main.DNA')
  def test_full_dna_workflow(self, mock_dna_class):
    mock_dna = Mock()
    mock_dna.has_special_tokens = False
    mock_dna.encode.return_value = [0, 1, 2]
    mock_dna.decode.return_value = "ATGTGCGCA"
    mock_dna.tokenize.return_value = ["ATG", "TGC", "GCA"]
    mock_dna.vocab_size = 64
    mock_dna_class.return_value = mock_dna
    
    tokenizer = Tokenizer("dna", 3)
    sequence = "ATGTGCGCA"
    
    # Test full encode-decode cycle
    encoded = tokenizer.encode(sequence)
    decoded = tokenizer.decode(encoded)
    tokens = tokenizer.tokenize(sequence)
    
    assert encoded == [0, 1, 2]
    assert decoded == sequence
    assert tokens == ["ATG", "TGC", "GCA"]
    assert tokenizer.vocab_size == 64

  @patch('biosaic._main.Protein')
  def test_protein_with_special_tokens(self, mock_protein_class):
    mock_protein = Mock()
    mock_protein.has_special_tokens = True
    mock_protein.encode.return_value = [1, 5, 10, 2]  # <s>, AA, GG, </s>
    mock_protein.decode.return_value = "<s>AAGG</s>"
    mock_protein_class.return_value = mock_protein
    
    tokenizer = Tokenizer("protein", 2, special_tokens=['<s>', '</s>'])
    sequence = "<s>AAGG</s>"
    
    encoded = tokenizer.encode(sequence)
    decoded = tokenizer.decode(encoded)
    
    assert encoded == [1, 5, 10, 2]
    assert decoded == sequence
    assert tokenizer.encoding == "protein/special_2k"

if __name__ == "__main__":
  pytest.main([__file__, "-v"])