# Biosaic Tokenizer Documentation

## Overview

Biosaic is a specialized tokenizer library designed for biological sequences, supporting both DNA and protein sequence tokenization. It provides k-mer based tokenization with support for continuous (sliding-window) and non-continuous (fixed-chunk) modes, along with special token handling for advanced biological sequence processing.

## Features

- **Dual Mode Support**: DNA and protein sequence tokenization
- **Flexible K-mer Sizes**: Up to 8-mers for DNA, up to 4-mers for proteins
- **Tokenization Modes**: Continuous (overlapping) and non-continuous (fixed-chunk)
- **Special Tokens**: Support for sequence markup and control tokens
- **Remote Vocabularies**: Pre-trained vocabularies loaded from remote repositories
- **Comprehensive Methods**: Encoding, decoding, tokenization, and biological utilities

## Installation

```python
from biosaic import Tokenizer
```

## Quick Start

### Basic DNA Tokenization

```python
# Initialize DNA tokenizer with 3-mers
dna_tokenizer = Tokenizer(mode="dna", kmer=3, continuous=True)

# Tokenize a DNA sequence
sequence = "ATCGATCG"
tokens = dna_tokenizer.tokenize(sequence)
print(tokens)  # ['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG']

# Encode to IDs
ids = dna_tokenizer.encode(sequence)
print(ids)  # [45, 89, 67, 123, 45, 89]

# Decode back to sequence
decoded = dna_tokenizer.decode(ids)
print(decoded)  # "ATCGATCG"
```

### Basic Protein Tokenization

```python
# Initialize protein tokenizer with 2-mers
protein_tokenizer = Tokenizer(mode="protein", kmer=2, continuous=True)

# Tokenize a protein sequence
sequence = "MKVLWAALLVTFLAGC"
tokens = protein_tokenizer.tokenize(sequence)
print(tokens)  # ['MK', 'KV', 'VL', 'LW', 'WA', ...]

# Get vocabulary size
print(protein_tokenizer.vocab_size)  # 441 (21^2 amino acids)
```

## Class Reference

### Tokenizer

The main tokenizer class that handles both DNA and protein sequences.

#### Constructor

```python
Tokenizer(mode: str, kmer: int, continuous: bool = False, special_tokens = None)
```

**Parameters:**

- `mode` (str): Sequence type - "dna" or "protein"
- `kmer` (int): K-mer length (max 8 for DNA, max 4 for protein)
- `continuous` (bool): Tokenization mode
  - `True`: Sliding-window (overlapping k-mers)
  - `False`: Fixed-chunk (non-overlapping k-mers)
- `special_tokens` (list/None/False): Special token configuration
  - `None`: Default tokens `['<S>', '</S>', '<P>', '<C>', '<M>']` (only with continuous=False)
  - `list`: Custom special tokens
  - `False`: No special tokens

**Raises:**

- `AssertionError`: Invalid mode or k-mer size
- `ValueError`: Special tokens used with continuous=True

#### Core Methods

##### encode(sequence: str) → List[int]

Converts biological sequence to integer token IDs.

```python
dna_tokenizer = Tokenizer("dna", kmer=3, continuous=True)
ids = dna_tokenizer.encode("ATCGAT")
print(ids)  # [45, 89, 67, 123]
```

##### decode(ids: List[int]) → str

Converts token IDs back to original sequence.

```python
sequence = dna_tokenizer.decode([45, 89, 67, 123])
print(sequence)  # "ATCGAT"
```

##### tokenize(sequence: str) → List[str]

Splits sequence into k-mer tokens.

```python
tokens = dna_tokenizer.tokenize("ATCGAT")
print(tokens)  # ['ATC', 'TCG', 'CGA', 'GAT']
```

##### detokenize(tokens: List[str]) → str

Reconstructs sequence from k-mer tokens.

```python
sequence = dna_tokenizer.detokenize(['ATC', 'TCG', 'CGA', 'GAT'])
print(sequence)  # "ATCGAT"
```

#### Biological Utility Methods

##### one_hot(sequence: str) → numpy.ndarray

Creates one-hot encoding matrix.

```python
dna_tokenizer = Tokenizer("dna", kmer=2, continuous=False)
one_hot_matrix = dna_tokenizer.one_hot("ATCG")
print(one_hot_matrix.shape)  # (2, vocab_size)
```

##### reverse_complement(sequence: str) → str

Returns reverse complement of DNA sequence (DNA only).

```python
dna_tokenizer = Tokenizer("dna", kmer=3)
rev_comp = dna_tokenizer.reverse_complement("ATCG")
print(rev_comp)  # "CGAT"
```

##### pad_sequence(sequence: str, target_length: int, pad_char: str = "-") → str

Pads sequence to target length.

```python
padded = dna_tokenizer.pad_sequence("ATCG", 10)
print(padded)  # "ATCG------"
```

#### Properties

##### vocab_size

Returns vocabulary size.

```python
print(dna_tokenizer.vocab_size)  # 625 (5^4 for 4-mer DNA)
```

##### vocab

Returns vocabulary dictionary mapping tokens to IDs.

```python
print(dna_tokenizer.vocab['ATG'])  # 45
```

## Tokenization Modes

### Continuous Mode (Sliding Window)

Generates overlapping k-mers using a sliding window approach.

```python
tokenizer = Tokenizer("dna", kmer=3, continuous=True)
tokens = tokenizer.tokenize("ATCGATCG")
print(tokens)  # ['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG']
```

**Use Cases:**

- Preserving sequence continuity
- Maximum information retention
- Machine learning applications requiring dense representation

### Non-Continuous Mode (Fixed Chunks)

Splits sequence into non-overlapping k-mer chunks.

```python
tokenizer = Tokenizer("dna", kmer=3, continuous=False)
tokens = tokenizer.tokenize("ATCGATCG")
print(tokens)  # ['ATC', 'GAT', 'CG']
```

**Use Cases:**

- Memory-efficient processing
- When overlapping information is not needed
- Compatible with special tokens

## Special Tokens

Special tokens provide sequence markup and control capabilities. They only work with `continuous=False`.

### Default Special Tokens

- `<S>`: Start of sequence
- `</S>`: End of sequence
- `<P>`: Padding token
- `<C>`: Control token
- `<M>`: Mask token

### Usage Example

```python
# Initialize with special tokens
tokenizer = Tokenizer("dna", kmer=3, continuous=False, special_tokens=['<S>', '</S>'])

# Use special tokens in sequence
sequence = "<S>ATCGATCG</S>"
tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['<S>', 'ATC', 'GAT', 'CG', '</S>']

# Encode with special tokens
ids = tokenizer.encode(sequence)
decoded = tokenizer.decode(ids)
print(decoded)  # "<S>ATCGATCG</S>"
```

### Custom Special Tokens

```python
custom_tokens = ['<START>', '<END>', '<GENE>', '<PROMOTER>']
tokenizer = Tokenizer("dna", kmer=2, continuous=False, special_tokens=custom_tokens)
```

## Advanced Usage Examples

### Processing Multiple Sequences

```python
def process_sequences(sequences, mode="dna", kmer=3):
    tokenizer = Tokenizer(mode=mode, kmer=kmer, continuous=True)
    
    results = []
    for seq in sequences:
        try:
            tokens = tokenizer.tokenize(seq)
            ids = tokenizer.encode(seq)
            results.append({
                'sequence': seq,
                'tokens': tokens,
                'ids': ids,
                'length': len(tokens)
            })
        except ValueError as e:
            print(f"Error processing {seq}: {e}")
    
    return results

# Example usage
dna_sequences = ["ATCGATCG", "GCTAGCTA", "TTAACCGG"]
results = process_sequences(dna_sequences)
```

### Batch Processing with Padding

```python
def create_padded_batch(sequences, tokenizer, max_length=None):
    # Tokenize all sequences
    all_tokens = [tokenizer.tokenize(seq) for seq in sequences]
    
    # Find maximum length if not specified
    if max_length is None:
        max_length = max(len(tokens) for tokens in all_tokens)
    
    # Pad and encode
    batch_ids = []
    for tokens in all_tokens:
        # Truncate or pad token list
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend(['<P>'] * (max_length - len(tokens)))
        
        # Convert to IDs
        ids = [tokenizer.vocab.get(token, 0) for token in tokens]
        batch_ids.append(ids)
    
    return batch_ids

# Example
tokenizer = Tokenizer("protein", kmer=2, continuous=False, special_tokens=['<P>'])
sequences = ["MKVLWAALL", "VTFLAGC", "ACDEFGHIKLMNPQRSTVWY"]
batch = create_padded_batch(sequences, tokenizer, max_length=10)
```

### Sequence Analysis

```python
def analyze_sequence_composition(sequence, tokenizer):
    tokens = tokenizer.tokenize(sequence)
    
    # Count token frequencies
    from collections import Counter
    token_counts = Counter(tokens)
    
    # Calculate statistics
    total_tokens = len(tokens)
    unique_tokens = len(token_counts)
    most_common = token_counts.most_common(5)
    
    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'vocabulary_coverage': unique_tokens / tokenizer.vocab_size,
        'most_common_tokens': most_common,
        'token_distribution': dict(token_counts)
    }

# Example analysis
dna_tokenizer = Tokenizer("dna", kmer=3, continuous=True)
sequence = "ATCGATCGATCGATCG" * 10  # Longer sequence for analysis
analysis = analyze_sequence_composition(sequence, dna_tokenizer)
print(f"Vocabulary coverage: {analysis['vocabulary_coverage']:.2%}")
```

## Error Handling

### Common Errors and Solutions

#### Invalid Characters

```python
try:
    tokenizer = Tokenizer("dna", kmer=3)
    tokenizer.encode("ATCXGATCG")  # X is invalid
except ValueError as e:
    print(f"Invalid character error: {e}")
    # Solution: Clean sequence or use valid DNA bases (A, T, G, C, -)
```

#### Special Tokens with Continuous Mode

```python
try:
    tokenizer = Tokenizer("dna", kmer=3, continuous=True, special_tokens=['<S>'])
except ValueError as e:
    print(f"Configuration error: {e}")
    # Solution: Use continuous=False with special tokens
```

#### K-mer Size Limits

```python
try:
    tokenizer = Tokenizer("protein", kmer=5)  # Too large
except AssertionError as e:
    print(f"K-mer size error: {e}")
    # Solution: Use kmer <= 4 for proteins, kmer <= 8 for DNA
```

## Performance Considerations

### Memory Usage

- **Continuous mode**: Higher memory usage due to overlapping tokens
- **Non-continuous mode**: More memory efficient
- **Large k-mers**: Exponentially larger vocabularies

### Speed Optimization

```python
# Pre-load tokenizer for repeated use
tokenizer = Tokenizer("dna", kmer=4, continuous=True)

# Batch process sequences
sequences = ["ATCG" * 100 for _ in range(1000)]
start_time = time.time()

all_ids = [tokenizer.encode(seq) for seq in sequences]

print(f"Processed {len(sequences)} sequences in {time.time() - start_time:.2f}s")
```

## Integration Examples

### With Machine Learning

```python
import numpy as np

def prepare_ml_data(sequences, tokenizer, max_length=100):
    # Tokenize and encode
    encoded_sequences = []
    for seq in sequences:
        ids = tokenizer.encode(seq)
        # Pad or truncate
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids.extend([0] * (max_length - len(ids)))  # 0 as padding ID
        encoded_sequences.append(ids)
    
    return np.array(encoded_sequences)

# Usage with neural networks
tokenizer = Tokenizer("dna", kmer=3, continuous=True)
sequences = ["ATCGATCG", "GCTAGCTA", "TTAACCGG"]
X = prepare_ml_data(sequences, tokenizer)
print(f"Input shape for ML model: {X.shape}")
```

### With Bioinformatics Pipelines

```python
def process_fasta_like_data(sequence_data, mode="dna", kmer=3):
    tokenizer = Tokenizer(mode=mode, kmer=kmer, continuous=False, 
                         special_tokens=['<S>', '</S>'])
    
    processed_data = []
    for seq_id, sequence in sequence_data.items():
        # Add sequence markers
        marked_sequence = f"<S>{sequence}</S>"
        
        # Tokenize and encode
        tokens = tokenizer.tokenize(marked_sequence)
        ids = tokenizer.encode(marked_sequence)
        
        processed_data.append({
            'id': seq_id,
            'original': sequence,
            'tokens': tokens,
            'encoded': ids,
            'length': len(sequence)
        })
    
    return processed_data

# Example usage
sequences = {
    'seq1': 'ATCGATCGATCG',
    'seq2': 'GCTAGCTAGCTA',
    'seq3': 'TTAACCGGTTAA'
}
processed = process_fasta_like_data(sequences)
```

## Best Practices

### Choosing K-mer Size

- **Small k-mers (2-3)**: Better for short sequences, higher generalization
- **Large k-mers (4-8)**: More specific, better for pattern recognition
- **Consider sequence length**: k-mer should be much smaller than typical sequence length

### Mode Selection

- **Use continuous=True for**:
  - Machine learning applications
  - When sequence context is important
  - Pattern recognition tasks

- **Use continuous=False for**:
  - Memory-constrained environments
  - When special tokens are needed
  - Simple sequence analysis

### Special Token Strategy

- Use meaningful special tokens for your application
- Keep special tokens short and distinctive
- Consider vocabulary size impact
- Document special token meanings for reproducibility

## Troubleshooting

### Common Issues

1. **Vocabulary loading fails**: Check internet connection and URL accessibility
2. **Memory errors with large sequences**: Use smaller k-mers or non-continuous mode
3. **Inconsistent results**: Ensure same tokenizer configuration across sessions
4. **Special tokens not working**: Verify continuous=False is set

### Debug Information

The library includes debug output for vocabulary loading:

```bash
DEBUGG INFO[104] [Saved] Vocabulary saved to /path/to/vocab.model
```

This helps verify that vocabularies are being loaded correctly from remote sources.

## Version Information

This documentation covers the Biosaic tokenizer library with support for:

- DNA sequences (bases: A, T, G, C, -)
- Protein sequences (20 standard amino acids + gap character)
- Remote vocabulary loading from GitHub repositories
- K-mer sizes up to 8 for DNA and 4 for proteins
- Flexible tokenization modes and special token support