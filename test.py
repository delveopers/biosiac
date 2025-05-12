from biosaic._dna import DNA
string = "ATGCCCCAACTAAATACCGCCGTATGACCCACCATAATTACCCCCATACTCCTGACACTATTTCTCGTCACCCAACTAAAAATATTAAATTCAAATTACCATCTACCCCCCTCACCAAAACCCATAAAAATAAAAAACTACAATAAACCCTGAGAACCAAAATGAACGAAAATCTATTCGCTTCATTCGCTGCCCCCACAATCCTAG"

token = DNA(kmer=4, continuous=False)
token.build_vocab()
cont_token = DNA(kmer=4, continuous=True)
cont_token.build_vocab()

tokenized = token.tokenize(string)
cont_tokenized = cont_token.tokenize(string)

detokenized = token.detokenize(ids=tokenized)
cont_detokenized = cont_token.detokenize(ids=cont_tokenized)

print("string len: ", len(string))
print(tokenized[:10])
print("norm len: ", len(tokenized))
print(cont_tokenized[:10])
print("cont len: ", len(cont_tokenized))


encoded = token.encode(string)
cont_encoded = cont_token.encode(string)

decoded = token.decode(encoded)
cont_decoded = cont_token.decode(cont_encoded)

print("cont: ", token.vocab_size)
print("norm: ", cont_token.vocab_size)

print("cont: ", cont_decoded[:-10])
print("norm: ", decoded[:-10])

print("cont: ", cont_detokenized == string)
print("norm: ", detokenized == string)

print("cont: ", cont_decoded == string)
print("norm: ", decoded == string)