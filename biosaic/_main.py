from ._dna import DNA
from ._protein import Protein

class Tokenizer:
  def __init__(self, mode: str, kmer: int, continuous: bool=False):
    assert (mode == "dna" or mode == "protein"), "Unknow mode type, choose b/w ``dna`` & ``protein``"
    assert (kmer <= 8), "KMer size supported only till 8!"
    if mode == "dna":
      self.tokenizer = DNA(kmer=kmer, continuous=continuous)
    else:
      self.tokenizer = Protein(kmer=kmer, continuous=continuous)

  def load(self):
    pass