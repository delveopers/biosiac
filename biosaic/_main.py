from ._dna import DNA
from ._protein import Protein
from typing import List

main_base_url = "https://raw.githubusercontent.com/delveopers/biosaic/main/vocab/"
dev_base_url = "https://raw.githubusercontent.com/delveopers/biosaic/dev/vocab/"
hugginface_url = "https://huggingface.co/shivendrra/BiosaicTokenizer/resolve/main/kmers/"

class Tokenizer:
  """
    Biosaic Tokenizer class for DNA and Protein sequences with special token support.

    This class wraps around DNA and Protein tokenizers, allowing encoding,
    decoding, tokenization, and detokenization of biological sequences 
    using pre-trained vocabularies stored remotely, with optional special tokens.

    Attributes:
      kmer (int): The k-mer size used for tokenization.
      continuous (bool): Whether to use a sliding-window tokenization (`True`) or fixed-length non-overlapping (`False`).
      special_tokens (list or None): List of special tokens to include in vocabulary.
      encoding (str): The encoding identifier used to locate and load vocab files.
      encoding_path (str): URL path pointing to the pretrained vocabulary model file.
      _tokenizer (DNA or Protein): Internal tokenizer instance specific to the sequence type.
  """
  def __init__(self, mode: str, kmer: int, continuous: bool=False, special_tokens=None):
    """
      Initializes the Tokenizer with the specified mode, k-mer length, tokenization style, and special tokens.

      Args:
        mode (str): Type of sequence to tokenize. Should be either "dna" or "protein".
        kmer (int): The k-mer length used for tokenization. Maximum allowed is 8 for DNA and 4 for protein.
        continuous (bool): If True, enables sliding-window tokenization (i.e., overlapping k-mers).
                          If False, tokenizes in fixed non-overlapping k-mer chunks.
        special_tokens (list or None): List of special tokens to add to vocabulary. 
                                      If None, uses default ['<s>', '</s>', '<p>', '<c>', '<m>'].
                                      If False, no special tokens are added.

      Raises:
        AssertionError: If an invalid mode is specified or k-mer size is above supported limit.
        ValueError: If special tokens are used with continuous=True.
    """
    assert (mode == "dna" or mode == "protein"), "Unknown mode type, choose b/w ``dna`` & ``protein``"
    if mode == "protein":
      assert (kmer <= 4), "KMer size supported only till 4 for protein!"
    else:
      assert (kmer <= 8), "KMer size supported only till 8 for DNA!"
    self.kmer, self.continuous, self.special_tokens = kmer, continuous, special_tokens
    
    if mode == "dna":
      self._tokenizer = DNA(kmer=kmer, continuous=continuous, special_tokens=special_tokens)
    else:
      self._tokenizer = Protein(kmer=kmer, continuous=continuous, special_tokens=special_tokens)

    if continuous:
      self.encoding = f"{mode}/cont_{kmer}k"
    else:
      self.encoding = f"{mode}/base_{kmer}k"

    self.encoding_path = main_base_url + self.encoding + ".model"
    self._tokenizer.load(model_path=self.encoding_path)
    self.encoding = f"{mode}/special_{kmer}k" if self._tokenizer.has_special_tokens else self.encoding

  def encode(self, sequence: str) -> list[int]:
    """
      Encodes a biological sequence into integer token IDs.

      Args:
        sequence (str): DNA or protein sequence composed of valid characters and/or special tokens.
      Returns:
        List[int]: Encoded token IDs corresponding to k-mers and special tokens in the sequence.
      Raises:
        ValueError: If the input sequence contains invalid characters.
    """
    return self._tokenizer.encode(sequence)
  
  def decode(self, ids: list[int]) -> str:
    """
      Decodes a list of token IDs back into the original sequence.

      Args:
        ids (List[int]): Encoded token IDs representing a biological sequence.
      Returns:
        str: Decoded DNA/protein sequence reconstructed from k-mers and special tokens.
    """
    return self._tokenizer.decode(ids)

  def tokenize(self, sequence: str) -> List[str]:
    """
      Splits the input biological sequence into k-mer tokens and special tokens.

      Args:
        sequence (str): DNA or protein string potentially containing special tokens.
      Returns:
        List[str]: List of k-mer substrings and special tokens, with variable lengths when special tokens present.
    """
    return self._tokenizer.tokenize(sequence)

  def detokenize(self, ids: List[str]) -> str:
    """
      Combines k-mer tokens and special tokens into the original sequence.

      Args:
        ids (List[str]): List of k-mer tokens and special tokens.
      Returns:
        str: Reconstructed sequence from tokenized substrings.
    """
    return self._tokenizer.detokenize(ids)

  def one_hot(self, sequence):
    return self._tokenizer.one_hot_encode(sequence)

  def reverse_complement(self, sequence):
    return self._tokenizer.reverse_complement(sequence)

  def pad_sequence(self, sequence, target_length, pad_char="-"):
    return self._tokenizer.pad_sequence(sequence, target_length, pad_char)

  @property
  def vocab_size(self):
    return self._tokenizer.vocab_size

  @property
  def vocab(self):
    return self._tokenizer.vocab

  def __str__(self):
    special_info = f", special_tokens={len(self.special_tokens) if self.special_tokens else 0}" if hasattr(self, 'special_tokens') else ""
    return f"biosaic.tokenizer <kmer={self.kmer}, encoding={self.encoding}, continuous={self.continuous}{special_info}>"