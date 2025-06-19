# Biosaic Tokenizer Documentation

A Python library for tokenizing DNA and protein sequences using k-mer based approaches, designed for bioinformatics and computational biology applications.

## Features

- **Dual Mode Support**: Works with both DNA and protein sequences
- **Flexible K-mer Sizes**: Support for k-mers up to 8 for DNA and 4 for proteins
- **Two Tokenization Modes**:
  - **Continuous**: Sliding-window approach with overlapping k-mers
  - **Non-continuous**: Fixed-length non-overlapping chunks
- **Remote Vocabulary Loading**: Automatically fetches pre-trained vocabularies
- **Comprehensive Encoding/Decoding**: Full round-trip support for sequences
- **Additional Utilities**: One-hot encoding, padding, reverse complement (DNA only)

## Installation

```python
# Import the tokenizer
from biosaic import Tokenizer
```

## Quick Start

### DNA Tokenization

```python
# Initialize DNA tokenizer with 3-mer, continuous mode
dna_tokenizer = Tokenizer(mode="dna", kmer=3, continuous=True)

# Example DNA sequence
sequence = "ATGCGATCGATC"

# Tokenize into k-mers
tokens = dna_tokenizer.tokenize(sequence)
print(tokens)  # ['ATG', 'TGC', 'GCG', 'CGA', 'GAT', 'ATC', 'TCG', 'CGA', 'GAT', 'ATC']

# Encode to integer IDs
encoded = dna_tokenizer.encode(sequence)
print(encoded)  # [124, 89, 67, ...]

# Decode back to sequence
decoded = dna_tokenizer.decode(encoded)
print(decoded)  # "ATGCGATCGATC"
```

### Protein Tokenization

```python
# Initialize protein tokenizer with 2-mer, continuous mode
protein_tokenizer = Tokenizer(mode="protein", kmer=2, continuous=True)

# Example protein sequence
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

# Tokenize into k-mers
tokens = protein_tokenizer.tokenize(sequence)
print(tokens[:5])  # ['MK', 'KT', 'TV', 'VR', 'RQ']

# Encode and decode
encoded = protein_tokenizer.encode(sequence)
decoded = protein_tokenizer.decode(encoded)
```

## API Reference

### Tokenizer Class

#### Constructor

```python
Tokenizer(mode: str, kmer: int, continuous: bool = False)
```

**Parameters:**
- `mode` (str): Sequence type - either "dna" or "protein"
- `kmer` (int): K-mer length (max 8 for DNA, max 4 for protein)
- `continuous` (bool): Tokenization mode
  - `True`: Sliding-window with overlapping k-mers
  - `False`: Fixed non-overlapping chunks

#### Core Methods

##### `tokenize(sequence: str) -> List[str]`
Splits sequence into k-mer tokens.

```python
tokenizer = Tokenizer("dna", kmer=3)
tokens = tokenizer.tokenize("ATGCGA")
# Returns: ['ATG', 'CGA'] (non-continuous) or ['ATG', 'TGC', 'GCG', 'CGA'] (continuous)
```

##### `detokenize(tokens: List[str]) -> str`
Reconstructs sequence from k-mer tokens.

```python
sequence = tokenizer.detokenize(['ATG', 'TGC', 'GCG'])
# Returns: "ATGCG" (continuous) or "ATGTGCGCG" (non-continuous)
```

##### `encode(sequence: str) -> List[int]`
Converts sequence to integer token IDs.

```python
ids = tokenizer.encode("ATGCGA")
# Returns: [45, 123, 67, 89] (example IDs)
```

##### `decode(ids: List[int]) -> str`
Converts integer IDs back to sequence.

```python
sequence = tokenizer.decode([45, 123, 67, 89])
# Returns: "ATGCGA"
```

#### Utility Methods

##### `one_hot(sequence: str) -> numpy.ndarray`
Creates one-hot encoding matrix for the sequence.

```python
one_hot_matrix = tokenizer.one_hot("ATGC")
# Returns: numpy array of shape (n_tokens, vocab_size)
```

##### `reverse_complement(sequence: str) -> str` (DNA only)
Returns reverse complement of DNA sequence.

```python
dna_tokenizer = Tokenizer("dna", kmer=3)
rev_comp = dna_tokenizer.reverse_complement("ATGC")
# Returns: "GCAT"
```

##### `pad_sequence(sequence: str, target_length: int, pad_char: str = "-") -> str`
Pads sequence to target length.

```python
padded = tokenizer.pad_sequence("ATGC", target_length=10, pad_char="N")
# Returns: "ATGCNNNNNN"
```

#### Properties

##### `vocab_size: int`
Returns the vocabulary size.

```python
print(tokenizer.vocab_size)  # e.g., 125 for DNA 3-mer continuous
```

##### `vocab: dict`
Returns the vocabulary mapping (k-mer -> ID).

```python
print(tokenizer.vocab)  # {'AAA': 0, 'AAT': 1, ...}
```

## Usage Examples

### Example 1: Basic DNA Analysis

```python
# Initialize tokenizer
tokenizer = Tokenizer("dna", kmer=4, continuous=True)

# Analyze a gene sequence
gene_sequence = "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACATCCC"

# Get k-mer tokens
kmers = tokenizer.tokenize(gene_sequence)
print(f"Number of 4-mers: {len(kmers)}")
print(f"First 5 k-mers: {kmers[:5]}")

# Encode for machine learning
encoded_sequence = tokenizer.encode(gene_sequence)
print(f"Encoded length: {len(encoded_sequence)}")

# Get reverse complement
rev_comp = tokenizer.reverse_complement(gene_sequence)
print(f"Reverse complement: {rev_comp}")
```

### Example 2: Protein Sequence Processing

```python
# Initialize protein tokenizer
tokenizer = Tokenizer("protein", kmer=3, continuous=False)

# Process protein sequence
protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

# Non-overlapping 3-mers
tokens = tokenizer.tokenize(protein)
print(f"Non-overlapping 3-mers: {tokens}")

# Create one-hot encoding for ML
one_hot = tokenizer.one_hot(protein)
print(f"One-hot shape: {one_hot.shape}")
```

### Example 3: Batch Processing

```python
def process_sequences(sequences, mode="dna", kmer=3):
    """Process multiple sequences efficiently."""
    tokenizer = Tokenizer(mode, kmer, continuous=True)
    
    results = []
    for seq in sequences:
        encoded = tokenizer.encode(seq)
        results.append({
            'sequence': seq,
            'encoded': encoded,
            'length': len(encoded),
            'vocab_coverage': len(set(encoded))
        })
    
    return results

# Example usage
dna_sequences = [
    "ATGCGATCGATCGAATGC",
    "GCATGCATGCATGCAT",
    "TTTTAAAACCCCGGGG"
]

results = process_sequences(dna_sequences)
for result in results:
    print(f"Sequence: {result['sequence'][:20]}...")
    print(f"Encoded length: {result['length']}")
    print(f"Unique tokens: {result['vocab_coverage']}")
    print("---")
```

### Example 4: Comparing Tokenization Modes

```python
sequence = "ATGCGATCGATCGAATGC"

# Continuous mode (overlapping)
continuous_tokenizer = Tokenizer("dna", kmer=3, continuous=True)
continuous_tokens = continuous_tokenizer.tokenize(sequence)

# Non-continuous mode (non-overlapping)
noncontinuous_tokenizer = Tokenizer("dna", kmer=3, continuous=False)
noncontinuous_tokens = noncontinuous_tokenizer.tokenize(sequence)

print(f"Original sequence: {sequence}")
print(f"Continuous tokens ({len(continuous_tokens)}): {continuous_tokens}")
print(f"Non-continuous tokens ({len(noncontinuous_tokens)}): {noncontinuous_tokens}")

# Reconstruct sequences
continuous_reconstructed = continuous_tokenizer.detokenize(continuous_tokens)
noncontinuous_reconstructed = noncontinuous_tokenizer.detokenize(noncontinuous_tokens)

print(f"Continuous reconstructed: {continuous_reconstructed}")
print(f"Non-continuous reconstructed: {noncontinuous_reconstructed}")
```

## Advanced Features

### Custom Vocabulary Paths

The tokenizer automatically loads pre-trained vocabularies from remote repositories. The vocabulary files are organized as:

- **Main branch**: `https://raw.githubusercontent.com/delveopers/biosaic/main/vocab/`
- **Dev branch**: `https://raw.githubusercontent.com/delveopers/biosaic/dev/vocab/`
- **HuggingFace**: `https://huggingface.co/shivendrra/BiosaicTokenizer/resolve/main/kmers/`

Vocabulary files follow the naming convention:
- Continuous: `{mode}/cont_{k}k.model`
- Non-continuous: `{mode}/base_{k}k.model`

### Error Handling

The tokenizer includes robust error handling:

```python
try:
    tokenizer = Tokenizer("dna", kmer=10)  # Too large
except AssertionError as e:
    print(f"Error: {e}")

try:
    tokenizer = Tokenizer("dna", kmer=3)
    tokenizer.encode("ATGXCGA")  # Invalid character
except ValueError as e:
    print(f"Error: {e}")
```

## Supported Characters

### DNA Sequences
- Standard bases: `A`, `T`, `G`, `C`
- Gap/padding: `-`

### Protein Sequences
- All 20 standard amino acids: `A`, `R`, `N`, `D`, `C`, `Q`, `E`, `G`, `H`, `I`, `L`, `K`, `M`, `F`, `P`, `S`, `T`, `W`, `Y`, `V`
- Gap/padding: `-`

## Performance Considerations

- **K-mer Size**: Larger k-mers create exponentially larger vocabularies
- **Continuous Mode**: Generates more tokens but preserves sequence information better
- **Memory Usage**: Vocabulary size grows as `alphabet_size^k` for continuous mode
- **Remote Loading**: First use requires internet connection to download vocabulary

## Best Practices

1. **Choose Appropriate K-mer Size**: Start with k=3 for DNA and k=2 for proteins
2. **Mode Selection**: Use continuous for sequence modeling, non-continuous for feature extraction
3. **Sequence Preprocessing**: Ensure sequences contain only valid characters
4. **Batch Processing**: Process multiple sequences together for efficiency
5. **Error Handling**: Always wrap tokenization in try-catch blocks for production code

## Troubleshooting

### Common Issues

1. **"Unknown mode type" Error**: Ensure mode is exactly "dna" or "protein"
2. **"KMer size supported only till X" Error**: Reduce k-mer size within supported limits
3. **"Invalid character" Error**: Check sequence for unsupported characters
4. **Network Issues**: Ensure internet connection for initial vocabulary download

### Debug Information

The tokenizer provides debug information during vocabulary loading and saving operations. Look for messages starting with "DEBUGG INFO" for troubleshooting.

## License and Attribution

This documentation covers the Biosaic Tokenizer library. Please refer to the original repository for license information and contribution guidelines.