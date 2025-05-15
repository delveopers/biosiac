# User Documentation

## Biosaic Tokenizer: Quick Start

### 1. Installation

Assuming you’ve published or cloned the repository, install via pip:

```bash
pip install biosaic
```

Or, in editable/development mode:

```bash
pip install -e biosaic
```

This will install the `biosaic` package and its runtime dependency (`requests`).

---

### 2. Package Layout

```text
biosaic/
├── __init__.py
├── _dna.py
├── _protein.py
└── _main.py   # defines Tokenizer
```

After installation, import from the top level:

```python
from biosaic import Tokenizer, DNA, Protein
```

***(If you prefer, you can also import the submodules directly.)***

---

### 3. Tokenizer Interface

The high‑level entry point is the `Tokenizer` class, which wraps both DNA and protein tokenizers.

```python
from biosaic import Tokenizer
```

#### 3.1 Initialization

```python
# mode: "dna" or "protein"
# kmer: integer length of each token (max 8 for DNA, 4 for protein)
# continuous: True  = overlapping (sliding window)
#             False = fixed non-overlapping chunks
tokenizer = Tokenizer(mode="dna", kmer=3, continuous=True)
```

* **`encoding_path`** is automatically set to fetch a pre‑trained vocabulary (remote `.model` file) and loaded under the hood.
* Raises `AssertionError` if you exceed supported k‑mer sizes or use an invalid mode.

#### 3.2 Basic Methods

```python
# 1) tokenize -> list of k‑mer substrings
tokens = tokenizer.tokenize("ATGCGTA")
# e.g. ["ATG", "TGC", "GCG", "CGT", "GTA"]

# 2) encode   -> integer IDs for each token
ids    = tokenizer.encode("ATGCGTA")
# e.g. [12, 47, 56, 33, 79]  (IDs depend on the loaded vocab)

# 3) decode   -> back to the original sequence
seq    = tokenizer.decode(ids)
# "ATGCGTA"

# 4) detokenize -> stitch tokens back
original = tokenizer.detokenize(tokens)
# "ATGCGTA"
```

#### 3.3 Inspecting the Vocabulary

```python
vocab_map = tokenizer.vocab
# e.g. { "ATG": 12, "TGC": 47, … }
```

---

### 4. Low‑Level Classes (Optional)

If you need more control—e.g. building your own vocab from scratch—you can use the `DNA` or `Protein` classes directly.

```python
from biosaic import DNA, Protein

dna = DNA(kmer=2, continuous=False)
dna.build_vocab()           # generate all possible 2‑mer combos
ids = dna.chars_to_ids(["AT", "GC"])
chars = dna.ids_to_chars(ids)
assert chars == ["AT", "GC"]
```

Similarly for proteins:

```python
prot = Protein(kmer=1, continuous=True)
prot.build_vocab()
prot.verify([0,1,2])        # returns matching info for overlapping k‑mers
```

Both classes support:

* `.tokenize(sequence)`
* `.detokenize(tokens)`
* `.encode(sequence)`
* `.decode(ids)`
* `.build_vocab()`
* `.chars_to_ids(tokens)`
* `.ids_to_chars(ids)`
* `.verify(...)`
* `.save(path, as_json=bool)`
* `.load(model_path)`

---

### 5. Example Script

```python
from biosaic import Tokenizer

def main():
  # Create a DNA tokenizer with 4‑mer sliding window
  tok = Tokenizer(mode="dna", kmer=4, continuous=True)

  # Tokenize & encode a sample sequence
  seq = "ATGCGTACGTA"
  tokens = tok.tokenize(seq)
  ids = tok.encode(seq)

  print("Tokens:", tokens)
  print("IDs:", ids)

  # Round‐trip decode
  print("Decoded:", tok.decode(ids))

if __name__ == "__main__":
  main()
```

Save as `example.py` and run:

```bash
python example.py
```
