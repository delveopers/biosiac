from biosaic._protein import Protein
string = "ATGCCCCAACTAAATACCGCCGTATGACCCACCATAATTACCCCCATACTCCTGACACTATTTCTCGTCACCCAACTAAAAATATTAAATTCAAATTACCATCTACCCCCCTCACCAAAACCCATAAAAATAAAAAACTACAATAAACCCTGAGAACCAAAATGAACGAAAATCTATTCGCTTCATTCGCTGCCCCCACAATCCTAG"

token = Protein(kmer=5, continuous=False)
token.build_vocab()
token.save("vocab/protein/base_5k")

cont_token = Protein(kmer=5, continuous=True)
cont_token.build_vocab()
cont_token.save("vocab/protein/cont_5k")