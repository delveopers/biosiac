from ._dna import DNA
from ._protein import Protein
from typing import List

main_base_url = "https://github.com/delveopers/biosaic/blob/main/model/"  # fetches from main branch
dev_base_url = "https://raw.githubusercontent.com/delveopers/biosaic/dev/model/"  # fetches from dev branch
hugginface_url = "https://huggingface.co/shivendrra/BiosaicTokenizer/resolve/main/kmers/"  # fetches from huggingface librrary

class Tokenizer:
  """
    Biosaic Tokenizer class for DNA and Protein sequences.

    This class wraps around DNA and Protein tokenizers, allowing encoding,
    decoding, tokenization, and detokenization of biological sequences 
    using pre-trained vocabularies stored remotely.

    Attributes:
      kmer (int): The k-mer size used for tokenization.
      continuous (bool): Whether to use a sliding-window tokenization (`True`) or fixed-length non-overlapping (`False`).
      encoding (str): The encoding identifier used to locate and load vocab files.
      encoding_path (str): URL path pointing to the pretrained vocabulary model file.
      _tokenizer (DNA or Protein): Internal tokenizer instance specific to the sequence type.
  """
  def __init__(self, mode: str, kmer: int, continuous: bool=False):
    """
      Initializes the Tokenizer with the specified mode, k-mer length, and tokenization style.

      Args:
        mode (str): Type of sequence to tokenize. Should be either "dna" or "protein".
        kmer (int): The k-mer length used for tokenization. Maximum allowed is 8.
        continuous (bool): If True, enables sliding-window tokenization (i.e., overlapping k-mers).
                           If False, tokenizes in fixed non-overlapping k-mer chunks.

      Raises:
        AssertionError: If an invalid mode is specified or k-mer size is above supported limit.
    """
    assert (mode == "dna" or mode == "protein"), "Unknow mode type, choose b/w ``dna`` & ``protein``"
    assert (kmer <= 8), "KMer size supported only till 8!"
    self.kmer, self.continuous = kmer, continuous
    if mode == "dna":
      self._tokenizer = DNA(kmer=kmer, continuous=continuous)
    else:
      self._tokenizer = Protein(kmer=kmer, continuous=continuous)
    if continuous:
      self.encoding = f"cont_{kmer}k"
    else:
      self.encoding = f"base_{kmer}k"
    self.encoding_path = dev_base_url + self.encoding + ".model"
    self._tokenizer.load(model_path=self.encoding_path)

  def encode(self, sequence: str) -> list[int]:
    """
      Encodes a biological sequence into integer token IDs.

      Args:
        sequence (str): DNA or protein sequence composed of valid characters.
      Returns:
        List[int]: Encoded token IDs corresponding to k-mers in the sequence.
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
        str: Decoded DNA/protein sequence reconstructed from k-mers.
    """
    return self._tokenizer.decode(ids)

  def tokenize(self, sequence: str) -> List[str]:
    """
      Splits the input biological sequence into k-mer tokens.

      Args:
        sequence (str): DNA or protein string.
      Returns:
        List[str]: List of k-mer substrings (tokens), overlapping or not based on `continuous`.
    """
    return self._tokenizer.tokenize(sequence)

  def detokenize(self, ids: List[str]) ->str:
    """
      Combines k-mer tokens into the original sequence.

      Args:
        ids (List[str]): List of k-mer tokens.
      Returns:
        str: Reconstructed sequence from tokenized substrings.
      """
    return self._tokenizer.detokenize(ids)

  @property
  def vocab(self):
    return self._tokenizer.vocab

  def __str__(self):
    return f"biosaic.tokenizer <kmer={self.kmer}, encoding={self.encoding}, continuous={self.continuous}>"