# Biosaic Tokenizer User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [DNA Tokenization](#dna-tokenization)
4. [Protein Tokenization](#protein-tokenization)
5. [Tokenization Modes](#tokenization-modes)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Common Use Cases](#common-use-cases)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

## Getting Started

### Installation
```python
# Assuming biosaic is installed
from biosaic import Tokenizer
```

### Quick Start Example
```python
# Create a DNA tokenizer with 3-mer k-mers
tokenizer = Tokenizer(mode="dna", kmer=3, continuous=True)

# Tokenize a DNA sequence
sequence = "ATGCGATCG"
tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['ATG', 'TGC', 'GCG', 'CGA', 'GAT', 'ATC', 'TCG']

# Encode to integer IDs
ids = tokenizer.encode(sequence)
print(ids)     # [64, 156, 99, 35, 71, 52, 152]

# Decode back to sequence
decoded = tokenizer.decode(ids)
print(decoded) # "ATGCGATCG"
```

## Basic Usage

### Creating a Tokenizer

The `Tokenizer` class is your main entry point. You need to specify three parameters:

```python
tokenizer = Tokenizer(mode, kmer, continuous)
```

**Parameters:**
- `mode` (str): Either `"dna"` or `"protein"`
- `kmer` (int): K-mer size (1-8 for DNA, 1-4 for protein)
- `continuous` (bool): Tokenization mode (default: False)

### Core Methods

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `tokenize()` | Split sequence into k-mer tokens | String sequence | List of k-mer strings |
| `detokenize()` | Reconstruct sequence from tokens | List of k-mer strings | Original sequence |
| `encode()` | Convert sequence to integer IDs | String sequence | List of integers |
| `decode()` | Convert integer IDs back to sequence | List of integers | Original sequence |

## DNA Tokenization

### Supported Characters
DNA sequences can contain: `A`, `T`, `G`, `C`, `-` (gap character)

### Basic DNA Examples

#### 2-mer DNA Tokenization
```python
# Non-continuous (chunked) tokenization
tokenizer = Tokenizer(mode="dna", kmer=2, continuous=False)
sequence = "ATGCTA"

tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['AT', 'GC', 'TA']

# Reconstruct original sequence
reconstructed = tokenizer.detokenize(tokens)
print(reconstructed)  # "ATGCTA"
```

#### 3-mer DNA Tokenization (Continuous)
```python
# Continuous (sliding window) tokenization
tokenizer = Tokenizer(mode="dna", kmer=3, continuous=True)
sequence = "ATGCTA"

tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['ATG', 'TGC', 'GCT', 'CTA']

# Encoding and decoding
ids = tokenizer.encode(sequence)
decoded = tokenizer.decode(ids)
print(f"Original: {sequence}")
print(f"Decoded:  {decoded}")  # Should match original
```

#### Working with Gaps
```python
tokenizer = Tokenizer(mode="dna", kmer=2, continuous=False)
sequence = "ATG-CTA"  # Sequence with gap

tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['AT', 'G-', 'CT', 'A']
```

### DNA K-mer Size Guidelines

| K-mer Size | Use Case | Vocabulary Size (Continuous) |
|------------|----------|------------------------------|
| 1 | Basic nucleotide analysis | 5 |
| 2 | Dinucleotide patterns | 25 |
| 3 | Codon analysis | 125 |
| 4 | Short motif detection | 625 |
| 5-6 | Gene expression analysis | 3,125 - 15,625 |
| 7-8 | Complex pattern recognition | 78,125 - 390,625 |

## Protein Tokenization

### Supported Characters
Protein sequences can contain the 20 standard amino acids plus gap character:
`A`, `R`, `N`, `D`, `C`, `Q`, `E`, `G`, `H`, `I`, `L`, `K`, `M`, `F`, `P`, `S`, `T`, `W`, `Y`, `V`, `-`

### Basic Protein Examples

#### 2-mer Protein Tokenization
```python
tokenizer = Tokenizer(mode="protein", kmer=2, continuous=True)
sequence = "ACDEFG"

tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['AC', 'CD', 'DE', 'EF', 'FG']

ids = tokenizer.encode(sequence)
print(f"Encoded IDs: {ids}")
```

#### 3-mer Protein Tokenization
```python
tokenizer = Tokenizer(mode="protein", kmer=3, continuous=False)
sequence = "ACDEFGHIK"

tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['ACD', 'EFG', 'HIK']

# Each token represents a triplet of amino acids
```

### Protein K-mer Size Guidelines

| K-mer Size | Use Case | Vocabulary Size (Continuous) |
|------------|----------|------------------------------|
| 1 | Amino acid composition | 21 |
| 2 | Dipeptide analysis | 441 |
| 3 | Tripeptide patterns | 9,261 |
| 4 | Complex structural motifs | 194,481 |

## Tokenization Modes

### Continuous Mode (Sliding Window)
**Best for:** Preserving sequence context, overlapping pattern analysis

```python
tokenizer = Tokenizer(mode="dna", kmer=3, continuous=True)
sequence = "ATGCAT"
tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['ATG', 'TGC', 'GCA', 'CAT']
```

**Advantages:**
- Maintains sequence continuity
- Better for ML models requiring context
- Captures overlapping patterns

**Disadvantages:**
- More tokens generated
- Higher memory usage
- Slower processing

### Non-Continuous Mode (Fixed Chunks)
**Best for:** Memory efficiency, non-overlapping feature extraction

```python
tokenizer = Tokenizer(mode="dna", kmer=3, continuous=False)
sequence = "ATGCAT"
tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['ATG', 'CAT']
```

**Advantages:**
- Memory efficient
- Faster processing
- Simpler token structure

**Disadvantages:**
- May lose sequence context
- End sequences might be truncated

## Advanced Features

### Vocabulary Access
```python
tokenizer = Tokenizer(mode="dna", kmer=2, continuous=True)

# Access the vocabulary
vocab = tokenizer.vocab
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample entries: {list(vocab.items())[:5]}")

# Check if a k-mer exists
if "AT" in vocab:
    print(f"'AT' has ID: {vocab['AT']}")
```

### Batch Processing
```python
def process_sequences(sequences, tokenizer):
    """Process multiple sequences efficiently"""
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
            results.append({'sequence': seq, 'error': str(e)})
    return results

# Example usage
sequences = ["ATGCGT", "CCGTAT", "AAATTT"]
tokenizer = Tokenizer(mode="dna", kmer=3, continuous=True)
results = process_sequences(sequences, tokenizer)
```

### Validation and Quality Control
```python
def validate_tokenization(sequence, tokenizer):
    """Validate that encoding/decoding preserves the original sequence"""
    try:
        # Test round-trip: sequence -> tokens -> sequence
        tokens = tokenizer.tokenize(sequence)
        reconstructed = tokenizer.detokenize(tokens)
        
        # Test round-trip: sequence -> ids -> sequence  
        ids = tokenizer.encode(sequence)
        decoded = tokenizer.decode(ids)
        
        return {
            'original': sequence,
            'tokens_roundtrip': reconstructed,
            'ids_roundtrip': decoded,
            'tokens_match': sequence == reconstructed,
            'ids_match': sequence == decoded,
            'num_tokens': len(tokens),
            'num_ids': len(ids)
        }
    except Exception as e:
        return {'error': str(e)}

# Example validation
sequence = "ATGCGATCG"
tokenizer = Tokenizer(mode="dna", kmer=3, continuous=True)
validation = validate_tokenization(sequence, tokenizer)
print(validation)
```

## Best Practices

### Choosing K-mer Size
1. **Start Small**: Begin with k=2 or k=3 for initial experiments
2. **Consider Context**: Larger k-mers capture more context but create larger vocabularies
3. **Memory Constraints**: Monitor memory usage with large k-mers
4. **Biological Relevance**: Choose sizes meaningful for your analysis (e.g., k=3 for codons)

### Choosing Tokenization Mode
1. **Use Continuous Mode When**:
   - Training ML models that need sequence context
   - Analyzing overlapping patterns
   - Sequence similarity analysis

2. **Use Non-Continuous Mode When**:
   - Memory is limited
   - Processing speed is critical
   - Analyzing independent features

### Error Handling
```python
def safe_tokenize(sequence, mode, kmer):
    """Safely tokenize with proper error handling"""
    try:
        tokenizer = Tokenizer(mode=mode, kmer=kmer, continuous=True)
        return tokenizer.tokenize(sequence)
    except ValueError as e:
        print(f"Invalid sequence: {e}")
        return None
    except AssertionError as e:
        print(f"Invalid parameters: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
tokens = safe_tokenize("ATGCXYZ", "dna", 3)  # Handles invalid 'X' character
```

## Common Use Cases

### 1. Gene Sequence Analysis
```python
# Analyze codon usage in coding sequences
tokenizer = Tokenizer(mode="dna", kmer=3, continuous=False)

gene_sequence = "ATGAAACGCATTAGCAAATTCGATCTAG"
codons = tokenizer.tokenize(gene_sequence)
print(f"Codons: {codons}")

# Count codon frequencies
from collections import Counter
codon_counts = Counter(codons)
print(f"Most common codons: {codon_counts.most_common(3)}")
```

### 2. Protein Domain Analysis
```python
# Analyze protein sequences for structural motifs
tokenizer = Tokenizer(mode="protein", kmer=4, continuous=True)

protein_seq = "MGSSHHHHHHSSGLVPRGSHMACDEFGHIKLMNPQRSTVWY"
tetrapeptides = tokenizer.tokenize(protein_seq)

# Look for specific motifs
motif = "HHHH"  # His-tag motif
if motif in tetrapeptides:
    print(f"Found His-tag motif at position: {tetrapeptides.index(motif)}")
```

### 3. Sequence Preprocessing for ML
```python
def prepare_ml_dataset(sequences, mode="dna", kmer=4):
    """Prepare sequences for machine learning"""
    tokenizer = Tokenizer(mode=mode, kmer=kmer, continuous=True)
    
    dataset = []
    for seq in sequences:
        try:
            ids = tokenizer.encode(seq)
            dataset.append(ids)
        except ValueError:
            print(f"Skipping invalid sequence: {seq[:20]}...")
            continue
    
    return dataset, tokenizer

# Example usage
sequences = ["ATGCGATCGATCG", "CCGATCGATCGAT", "TTACGATCGATCG"]
ml_data, tokenizer = prepare_ml_dataset(sequences)
print(f"Prepared {len(ml_data)} sequences for ML")
```

### 4. Sequence Comparison
```python
def compare_sequences(seq1, seq2, kmer=3):
    """Compare two sequences using k-mer analysis"""
    tokenizer = Tokenizer(mode="dna", kmer=kmer, continuous=True)
    
    tokens1 = set(tokenizer.tokenize(seq1))
    tokens2 = set(tokenizer.tokenize(seq2))
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    jaccard_similarity = len(intersection) / len(union) if union else 0
    
    return {
        'common_kmers': intersection,
        'unique_to_seq1': tokens1 - tokens2,
        'unique_to_seq2': tokens2 - tokens1,
        'jaccard_similarity': jaccard_similarity
    }

# Example
seq1 = "ATGCGATCG"
seq2 = "ATGCGTTCG" 
comparison = compare_sequences(seq1, seq2)
print(f"Similarity: {comparison['jaccard_similarity']:.2f}")
```

## Troubleshooting

### Common Errors

#### 1. Invalid Character Error
```
ValueError: Invalid character in DNA sequence
```
**Solution**: Check your sequence for invalid characters
```python
# Valid DNA characters: A, T, G, C, -
sequence = "ATGCXYZ"  # X, Y, Z are invalid
# Clean the sequence first
clean_sequence = ''.join(c for c in sequence if c in 'ATGC-')
```

#### 2. K-mer Size Error
```
AssertionError: KMer size supported only till 8 for DNA!
```
**Solution**: Use supported k-mer sizes
```python
# DNA: max k-mer = 8
# Protein: max k-mer = 4
tokenizer = Tokenizer(mode="dna", kmer=8)  # OK
tokenizer = Tokenizer(mode="dna", kmer=9)  # Error
```

#### 3. Empty Sequence
```python
# Handle empty sequences
def safe_process(sequence, tokenizer):
    if not sequence:
        return []
    return tokenizer.tokenize(sequence)
```

#### 4. Network Issues (Remote Loading)
```
RuntimeError: Failed to download model from URL
```
**Solution**: Check internet connection or use local vocabulary building

### Performance Issues

#### Memory Usage
```python
# Monitor memory usage with large vocabularies
import psutil
import os

def check_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory before: {check_memory_usage():.2f} MB")
tokenizer = Tokenizer(mode="dna", kmer=8, continuous=True)  # Large vocab
print(f"Memory after: {check_memory_usage():.2f} MB")
```

#### Processing Speed
```python
import time

def benchmark_tokenization(sequence, kmer_sizes):
    """Benchmark different k-mer sizes"""
    results = {}
    for kmer in kmer_sizes:
        tokenizer = Tokenizer(mode="dna", kmer=kmer, continuous=True)
        
        start_time = time.time()
        tokens = tokenizer.tokenize(sequence)
        end_time = time.time()
        
        results[kmer] = {
            'time': end_time - start_time,
            'num_tokens': len(tokens)
        }
    return results

# Example benchmark
long_sequence = "ATGC" * 1000  # 4000 bp sequence
results = benchmark_tokenization(long_sequence, [2, 3, 4, 5])
for kmer, stats in results.items():
    print(f"K={kmer}: {stats['time']:.4f}s, {stats['num_tokens']} tokens")
```

## API Reference

### Tokenizer Class

#### Constructor
```python
Tokenizer(mode: str, kmer: int, continuous: bool = False)
```

#### Methods

**tokenize(sequence: str) → List[str]**
- Splits sequence into k-mer tokens
- Returns list of k-mer strings

**detokenize(tokens: List[str]) → str**
- Reconstructs sequence from k-mer tokens
- Returns original sequence string

**encode(sequence: str) → List[int]**
- Converts sequence to integer token IDs
- Returns list of integer IDs

**decode(ids: List[int]) → str**
- Converts integer IDs back to sequence
- Returns reconstructed sequence string

#### Properties

**vocab** → Dict[str, int]
- Access to the vocabulary mapping
- Returns dictionary of k-mer → ID mappings

**kmer** → int
- K-mer size used by tokenizer

**continuous** → bool  
- Tokenization mode (True = continuous, False = chunked)

**encoding** → str
- Encoding identifier string

### Error Types

- `ValueError`: Invalid characters in sequence
- `AssertionError`: Invalid parameters (mode, k-mer size)
- `RuntimeError`: Network or file loading errors
- `TypeError`: Invalid file format or data types

---

This user guide provides comprehensive instructions for using the Biosaic tokenizer effectively. For technical details about the implementation, refer to the Technical Documentation.
