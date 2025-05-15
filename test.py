from biosaic._dna import DNA
string = "ATGCCCCAACTAAATACCGCCGTATGACCCACCATAATTACCCCCATACTCCTGACACTATTTCTCGTCACCCAACTAAAAATATTAAATTCAAATTACCATCTACCCCCCTCACCAAAACCCATAAAAATAAAAAACTACAATAAACCCTGAGAACCAAAATGAACGAAAATCTATTCGCTTCATTCGCTGCCCCCACAATCCTAG"

token = DNA(kmer=5, continuous=False)
token.build_vocab()
token.save("vocab.as_json/base_5k", as_json=True)

cont_token = DNA(kmer=5, continuous=True)
cont_token.build_vocab()
cont_token.save("vocab.as_json/cont_5k", as_json=True)