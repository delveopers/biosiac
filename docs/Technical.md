# Biosaic Tokenizer Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [K-mer Tokenization Logic](#k-mer-tokenization-logic)
4. [Vocabulary Management](#vocabulary-management)
5. [Encoding/Decoding Pipeline](#encodingdecoding-pipeline)
6. [Remote Model Loading](#remote-model-loading)
7. [Data Structures](#data-structures)
8. [Error Handling](#error-handling)
9. [Performance Considerations](#performance-considerations)
10. [Extension Points](#extension-points)
11. [Known Issues & Limitations](#known-issues--limitations)

## Architecture Overview

The Biosaic tokenizer follows a **composition-based architecture** with three main layers:

```
Tokenizer (Facade Layer)
    ├── DNA (Sequence-Specific Implementation)
    └── Protein (Sequence-Specific Implementation)
```

### Design Patterns Used

- **Facade Pattern**: `Tokenizer` class provides a unified interface
- **Strategy Pattern**: Different tokenization strategies for DNA vs Protein
- **Template Method**: Common interface across sequence types

### Key Architectural Decisions

- **Remote Vocabulary Loading**: Vocabularies are fetched from remote URLs rather than bundled
- **Lazy Loading**: Vocabularies are loaded only when tokenizer is instantiated
- **Unified Interface**: Same API for both DNA and protein sequences

## Core Components

### 1. Tokenizer Class (`_main.py`)

**Purpose**: Main facade providing unified access to sequence-specific tokenizers

**Key Responsibilities**:

- Route requests to appropriate sequence-specific tokenizer
- Manage remote model loading
- Provide consistent API across sequence types

**Critical Attributes**:

```python
self.kmer: int              # K-mer size (1-8 for DNA, 1-4 for protein)
self.continuous: bool       # Tokenization mode
self.encoding: str         # Vocabulary identifier
self.encoding_path: str    # Remote model URL
self._tokenizer: DNA|Protein  # Actual implementation
```

### 2. DNA Class (`_dna.py`)

**Purpose**: DNA-specific tokenization implementation

**Alphabet**: `['A', 'T', 'G', 'C', '-']` (5 characters)
**Max K-mer Size**: 8 (vocabulary size = 5^8 = 390,625 for continuous mode)

### 3. Protein Class (`_protein.py`)

**Purpose**: Protein-specific tokenization implementation

**Alphabet**: 20 standard amino acids + 1 gap character = 21 total
**Max K-mer Size**: 4 (vocabulary size = 21^4 = 194,481 for continuous mode)

## K-mer Tokenization Logic

### Continuous Mode (Sliding Window)

```python
# For sequence "ATGC" with kmer=3:
# Returns: ["ATG", "TGC"]
def tokenize_continuous(sequence, kmer):
    return [sequence[i:i+kmer] for i in range(len(sequence) - kmer + 1)]
```

**Use Cases**: 

- Preserves local sequence context
- Better for overlapping pattern recognition
- Essential for maintaining sequence continuity in reconstruction

### Non-Continuous Mode (Fixed Chunks)

```python
# For sequence "ATGC" with kmer=3:
# Returns: ["ATG", "C"]
def tokenize_noncontinuous(sequence, kmer):
    return [sequence[i:i+kmer] for i in range(0, len(sequence), kmer)]
```

**Use Cases**:

- More memory efficient
- Faster processing
- Better for non-overlapping feature extraction

### Critical Implementation Detail: Detokenization

**Continuous Mode Reconstruction**:

```python
def detokenize_continuous(tokens):
    if not tokens:
        return ""
    # Take first character of each token + remaining chars of last token
    return "".join(token[0] for token in tokens) + tokens[-1][1:]
```

**Algorithm Logic**:

- For tokens `["ATG", "TGC", "GCA"]`
- Take: `A` + `T` + `G` + `CA` = `"ATGCA"`
- This works because overlapping k-mers share `k-1` characters

## Vocabulary Management

### Vocabulary Generation Strategy

#### Continuous Mode

```python
vocab_size = len(alphabet) ** kmer
# All possible k-mer combinations of fixed length
```

#### Non-Continuous Mode  
```python
vocab_size = sum(len(alphabet) ** i for i in range(1, kmer+1))
# All combinations from length 1 to k
```

### Vocabulary Structure
```python
{
    "vocab": {
        "A": 0, "T": 1, "G": 2, "C": 3,  # 1-mers
        "AA": 4, "AT": 5, ...             # 2-mers
        # ... up to k-mers
    },
    "ids_to_token": {0: "A", 1: "T", ...}  # Reverse mapping
}
```

### Build Process
1. **Generate Combinations**: Use `itertools.product()` for systematic generation
2. **Sort Alphabetically**: Ensures consistent vocabulary ordering
3. **Create Mappings**: Build both forward and reverse dictionaries
4. **Serialize**: Save as pickle (.model) or JSON (.json)

## Encoding/Decoding Pipeline

### Encoding Pipeline
```
Input Sequence → Validation → Tokenization → Vocabulary Lookup → Token IDs
```

1. **Validation**: Check for invalid characters
2. **Case Normalization**: Convert to uppercase
3. **Tokenization**: Apply k-mer splitting logic
4. **Mapping**: Convert k-mers to integer IDs via vocabulary

### Decoding Pipeline
```
Token IDs → Vocabulary Lookup → K-mer Tokens → Detokenization → Sequence
```

1. **ID to Token**: Map integers back to k-mer strings
2. **Detokenization**: Reconstruct sequence using mode-specific logic
3. **Validation**: Verify reconstruction integrity

### Critical Bug Risk Areas

**Encoding Issues**:
- Missing k-mer in vocabulary (partially built vocab)
- Invalid characters in input sequence
- Empty sequence handling

**Decoding Issues**:
- Invalid token IDs (out of vocabulary range)
- Incorrect detokenization for continuous mode
- Order-dependent reconstruction errors

## Remote Model Loading

### URL Structure
```python
main_base_url = "https://raw.githubusercontent.com/delveopers/biosaic/main/vocab/"
encoding_path = main_base_url + f"{mode}/{continuous_prefix}_{kmer}k.model"
```

### Loading Pipeline
1. **URL Construction**: Build path based on mode, k-mer size, continuity
2. **Download**: Use `urllib.request.urlretrieve()` to temporary file
3. **Deserialization**: Load pickle/JSON data
4. **Validation**: Verify vocabulary completeness
5. **Mapping Creation**: Build reverse lookup dictionary

### Error Handling
- **Network Failures**: Catch download exceptions
- **File Format Errors**: Handle corrupted vocabularies
- **Missing Models**: Fallback to local vocabulary building

## Data Structures

### Primary Data Structures
```python
class TokenizerState:
    vocab: Dict[str, int]           # K-mer → ID mapping
    ids_to_token: Dict[int, str]    # ID → K-mer mapping  
    vocab_size: int                 # Total vocabulary size
    kmer: int                       # K-mer length
    continuous: bool                # Tokenization mode
```

### Memory Considerations
- **DNA 8-mer continuous**: ~390K vocab entries = ~3-6MB memory
- **Protein 4-mer continuous**: ~194K vocab entries = ~1-3MB memory
- **Reverse mappings**: Double memory usage for bidirectional lookup

## Error Handling

### Input Validation
```python
# Character validation
if any(ch not in self._base_chars for ch in sequence):
    raise ValueError("Invalid character in [DNA/Protein] sequence")

# K-mer size validation  
assert (kmer <= 8), "KMer size supported only till 8 for DNA!"
assert (kmer <= 4), "KMer size supported only till 4 for protein!"
```

### Runtime Error Categories
1. **Vocabulary Errors**: Missing k-mers, corrupted files
2. **Network Errors**: Failed downloads, timeouts
3. **Input Errors**: Invalid sequences, wrong data types
4. **Memory Errors**: Large vocabulary loading failures

## Performance Considerations

### Time Complexity
- **Tokenization**: O(n) where n = sequence length
- **Encoding**: O(t) where t = number of tokens  
- **Decoding**: O(t) for token lookup + O(t) for reconstruction
- **Vocabulary Loading**: O(v) where v = vocabulary size

### Space Complexity
- **Vocabulary Storage**: O(v) for forward + O(v) for reverse = O(2v)
- **Tokenization**: O(t) temporary token storage
- **Sequence Reconstruction**: O(n) output sequence

### Optimization Opportunities
1. **Lazy Vocabulary Loading**: Load only needed k-mer sizes
2. **Vocabulary Compression**: Use more efficient serialization
3. **Batch Processing**: Process multiple sequences together
4. **Memory Mapping**: Use memory-mapped files for large vocabularies

## Extension Points

### Adding New Sequence Types
1. **Create New Class**: Follow DNA/Protein pattern
2. **Define Alphabet**: Set valid characters
3. **Implement Interface**: tokenize, detokenize, encode, decode methods
4. **Update Main Tokenizer**: Add mode routing logic

### Custom Tokenization Strategies
```python
class CustomTokenizer:
    def tokenize(self, sequence):
        # Implement custom logic (e.g., BPE, sentencepiece)
        pass
    
    def detokenize(self, tokens):
        # Implement reverse logic
        pass
```

### Vocabulary Customization
- **Custom Alphabets**: Modify `_base_chars` list
- **Special Tokens**: Add padding, unknown, start/end tokens
- **Hierarchical Vocabularies**: Multi-level k-mer encoding

## Known Issues & Limitations

### Limitations

1. **Fixed K-mer Sizes**: No dynamic k-mer selection
2. **No Ambiguous Bases**: No support for IUPAC ambiguous nucleotides
3. **Memory Usage**: Large vocabularies consume significant RAM
4. **No Streaming**: Must load entire vocabulary into memory

### Potential Improvements

1. **Streaming Tokenization**: Process sequences without full vocabulary loading
2. **Compressed Vocabularies**: Use trie structures or compressed mappings
3. **Better Error Messages**: More descriptive error information
4. **Validation Methods**: Built-in sequence validation utilities
5. **Performance Metrics**: Built-in timing and memory usage tracking

### Testing Recommendations

1. **Edge Cases**: Empty sequences, single characters, maximum k-mer sizes
2. **Round-trip Testing**: Encode → Decode should return original sequence
3. **Vocabulary Completeness**: Ensure all possible k-mers are in vocabulary
4. **Memory Testing**: Test with large sequences and vocabularies
5. **Network Testing**: Test remote loading with various network conditions

### Debugging Guidelines

1. **Vocabulary Issues**: Check `vocab` and `ids_to_token` consistency
2. **Tokenization Issues**: Verify k-mer generation logic
3. **Reconstruction Issues**: Test detokenization with known inputs
4. **Performance Issues**: Profile vocabulary loading and tokenization steps

---

This technical documentation serves as a comprehensive blueprint for understanding, maintaining, and extending the Biosaic tokenizer codebase. Developers should refer to this document when debugging issues, implementing new features, or optimizing performance.
