# from biosaic._protein import Protein
from biosaic import Tokenizer

# Test sequence
dna_seq = "ATCGATCGTTAAGGCC"
protein_seq = "MKLLVTACLVVVLLLSHAFPET"

print("=== DNA TOKENIZATION ===")

# 1. Default special tokens
print("\n1. Default Special Tokens:")
tokenizer1 = Tokenizer("dna", kmer=3, continuous=False)
seq_with_special = f"<S>{dna_seq}<M>GGGGAAAA</S>"
tokens1 = tokenizer1.tokenize(seq_with_special)
encoded1 = tokenizer1.encode(seq_with_special)
decoded1 = tokenizer1.decode(encoded1)
print(f"Sequence: {seq_with_special}")
print(f"Tokens: {tokens1}")
print(f"Encoded: {encoded1}")
print(f"Decoded: {decoded1}")
print(f"Vocab size: {tokenizer1.vocab_size}")

# 2. User-defined special tokens  
print("\n2. User-Defined Special Tokens:")
tokenizer2 = Tokenizer("dna", kmer=3, continuous=False, special_tokens=['<START>', '<END>', '<MASK>', '<CLS>'])
seq_with_custom = f"<START>{dna_seq}<MASK>TTTTCCCC<END>"
tokens2 = tokenizer2.tokenize(seq_with_custom)
encoded2 = tokenizer2.encode(seq_with_custom)
decoded2 = tokenizer2.decode(encoded2)
print(f"Sequence: {seq_with_custom}")
print(f"Special_tokens: {tokenizer2.special_tokens}")
print(f"Tokens: {tokens2}")
print(f"Encoded: {encoded2}")
print(f"Decoded: {decoded2}")
print(f"Vocab size: {tokenizer2.vocab_size}")

# 3. No special tokens
print("\n3. No Special Tokens:")
tokenizer3 = Tokenizer("dna", kmer=3, continuous=True)
tokens3 = tokenizer3.tokenize(dna_seq)
encoded3 = tokenizer3.encode(dna_seq)
decoded3 = tokenizer3.decode(encoded3)
print(f"Sequence: {dna_seq}")
print(f"Tokens: {tokens3}")
print(f"Encoded: {encoded3}")
print(f"Decoded: {decoded3}")
print(f"Vocab size: {tokenizer3.vocab_size}")

print("\n=== PROTEIN TOKENIZATION ===")

# 1. Default special tokens
print("\n1. Default Special Tokens:")
p_tokenizer1 = Tokenizer("protein", kmer=2, continuous=False)
p_seq_with_special = f"<S>{protein_seq}<P>AAA</S>"
p_tokens1 = p_tokenizer1.tokenize(p_seq_with_special)
p_encoded1 = p_tokenizer1.encode(p_seq_with_special)
p_decoded1 = p_tokenizer1.decode(p_encoded1)
print(f"Sequence: {p_seq_with_special}")
print(f"Tokens: {p_tokens1}")
print(f"Encoded: {p_encoded1}")
print(f"Decoded: {p_decoded1}")
print(f"Vocab size: {p_tokenizer1.vocab_size}")

# 2. User-defined special tokens
print("\n2. User-Defined Special Tokens:")
p_tokenizer2 = Tokenizer("protein", kmer=2, continuous=False, special_tokens=['<BEGIN>', '<STOP>'])
p_seq_with_custom = f"<BEGIN>{protein_seq}<STOP>"
p_tokens2 = p_tokenizer2.tokenize(p_seq_with_custom)
p_encoded2 = p_tokenizer2.encode(p_seq_with_custom)
p_decoded2 = p_tokenizer2.decode(p_encoded2)
print(f"Sequence: {p_seq_with_custom}")
print(f"Tokens: {p_tokens2}")
print(f"Encoded: {p_encoded2}")
print(f"Decoded: {p_decoded2}")
print(f"Vocab size: {p_tokenizer2.vocab_size}")

# 3. No special tokens
print("\n3. No Special Tokens:")
p_tokenizer3 = Tokenizer("protein", kmer=2, continuous=True)
p_tokens3 = p_tokenizer3.tokenize(protein_seq)
p_encoded3 = p_tokenizer3.encode(protein_seq)
p_decoded3 = p_tokenizer3.decode(p_encoded3)
print(f"Sequence: {protein_seq}")
print(f"Tokens: {p_tokens3}")
print(f"Encoded: {p_encoded3}")
print(f"Decoded: {p_decoded3}")
print(f"Vocab size: {p_tokenizer3.vocab_size}")

print("\n=== COMPARISON ===")
print(f"DNA vocab sizes: Default={tokenizer1.vocab_size}, Custom={tokenizer2.vocab_size}, None={tokenizer3.vocab_size}")
print(f"Protein vocab sizes: Default={p_tokenizer1.vocab_size}, Custom={p_tokenizer2.vocab_size}, None={p_tokenizer3.vocab_size}")