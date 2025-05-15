# Biosaic

## Overview

Biosaic(Bio-Mosaic) is a tokenizer library built for [Enigma2](https://github.com/shivendrra/enigma2). It contains: Tokenizer, Embedder for DNA & Amino Acid Protein Sequences. Has a VQ-VAE & Evoformer architecture based encoders that could convert sequences into embeddings and vice-versa for model training use-case.

## Features

- **Tokenization:** converts the sequences into K-Mers.
- **Encoding:** converts sequences into embeddings for classification, training purposes.
- **Easy use:** it's very basic and easy to use library.
- **SoTA encoder:** Evoformer & VQ-VAE model are inspired from the [AlphaFold-2](https://www.biorxiv.org/content/10.1101/2024.12.02.626366v1.full)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with support for GCC or Clang.
- **Python**: Version 3.9 or higher.

### Dependencies

- **Python Modules**:
  - `pickle`: for loading and saving model files.
  - `os`: for file and path handling.
  - `urllib`: for loading the vocabs from repo.
  - `tempfile`: for loading the vocabs from repo.

## Installation

#### 1. From PyPI:
  
  ```cmd
	pip install biosaic
  ```

#### 2. Clone the Repo:
  
  ```bash
  git clone https://github.com/delveopers/biosaic.git
  cd biosaic
  ```

## Usage

Create an instance of the tokenizer with a specified k-mer size, & split them into tokens, encode & decode them fastly:

```python
import biosaic
from biosaic import tokenizer

token = tokenizer(mode="dna", kmer=3, continuous=True)
print(token.vocab_size)

sequence = "TCTTACATAGAAAGGAGCGGTATTTGGTATGAATTTATTTGCAACTGACTG"
encoded = token.encode(sequence)
decoded = token.decode(encoded)
tokenized = token.tokenize(sequence)

print(tokenized)
print(encoded[:100])
print(decoded[:300])
print(decoded == sequence)
```

For more information refer to the docs:
- [Usage.md](https://github.com/delveopers/biosaic/blob/dev/docs/Usage.md)
- [Technical.md](https://github.com/delveopers/biosaic/blob/dev/docs/Technical.md)

#### ***Output***

```cmd
84

['TCT', 'CTT', 'TTA', 'TAC', 'ACA', 'CAT', 'ATA', 'TAG', 'AGA', 'GAA', 'AAA', 'AAG', 'AGG', 'GGA', 'GAG', 'AGC', 'GCG', 'CGG', 'GGT', 'GTA', 'TAT', 'ATT', 'TTT', 'TTG', 'TGG', 'GGT', 'GTA', 'TAT', 'ATG', 'TGA', 'GAA', 'AAT', 'ATT', 'TTT', 'TTA', 'TAT', 'ATT', 'TTT', 'TTG', 'TGC', 'GCA', 'CAA', 'AAC', 'ACT', 'CTG', 'TGA', 'GAC', 'ACT', 'CTG']

[75, 51, 80, 69, 24, 39, 32, 70, 28, 52, 20, 22, 30, 60, 54, 29, 58, 46, 63, 64, 71, 35, 83, 82, 78, 63, 64, 71, 34, 76, 52, 23, 35, 83, 80, 71, 35, 83, 82, 77, 56, 36, 21, 27, 50, 76, 53, 27, 50]

TCTTACATAGAAAGGAGCGGTATTTGGTATGAATTTATTTGCAACTGACTG

True
```