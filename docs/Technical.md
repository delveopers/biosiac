
# Biosaic Technical Specification

## 1. Overview

BioSAIC is a lightweight Python library for k‑mer tokenization of biological sequences (DNA & proteins). It loads pre‑trained vocabularies remotely and supports both sliding‑window (“continuous”) and fixed‑chunk (“non‑continuous”) tokenization modes.

## 2. Architecture & Modules

```text
biosaic/
├── _dna.py       # DNA-specific tokenizer
├── _protein.py   # Protein-specific tokenizer
└── _main.py      # Generic Tokenizer wrapper
```

1. **\_dna.py**

   * Class: `DNA`
   * Alphabet: `['A', 'T', 'G', 'C', '-']`
   * Configurable `kmer` (1–8) and `continuous` (True/False).
   * Supports vocab building, encode/decode, tokenize/detokenize, verification, save/load.

2. **\_protein.py**

   * Class: `Protein`
   * Alphabet: 21 amino acids + gap
   * Configurable `kmer` (1–4) and `continuous`.
   * Same API as `DNA`.

3. **\_main.py**

   * Class: `Tokenizer`
   * Chooses between `DNA` or `Protein` based on `mode`.
   * Computes `encoding_path` (dev vs. main branch URLs) and calls `.load()` on the underlying tokenizer.

## 3. Core Concepts

### 3.1 k‑mer Tokenization

* **Continuous (sliding window)**
  Produces overlapping k‑mers:

```text
  sequence: ATGC
  k=2 → ["AT", "TG", "GC"]
```

* **Non‑continuous**
  Fixed, non‑overlapping chunks:

```text
  sequence: ATGCGT
  k=2 → ["AT", "GC", "GT"]
```

### 3.2 Vocabulary & ID Mapping

* **`build_vocab()`**
  Enumerates all possible k‑mers (up to k) and assigns integer IDs.
* **`save(path, as_json)`**
  Persists `{kmer, vocab_size, trained_vocab}` to `.model` (pickle) or `.json`.
* **`load(model_path)`**

  * If `model_path` is a URL → downloads to a temp file.
  * Loads from `.model` or `.json`.
  * Reconstructs `vocab`, `vocab_size`, `kmer`, `ids_to_token`.

### 3.3 Verification

Both `DNA` and `Protein` implement:

```python
verify(ids: List[int] | List[str], file: Optional[str])
```

* Converts IDs → tokens if necessary.
* Checks that each adjacent pair of k‑mers overlaps correctly:

```python
match = kmer1[1:] == kmer2[:-1]
```

* Returns a list of `{kmer1, kmer2, match}`, and writes `verify.json` if `file` is provided.

## 4. Remote Model Hosting

* **`main_base_url`**: [https://github.com/delveopers/biosaic/blob/main/vocab/](https://github.com/delveopers/biosaic/blob/main/vocab/)
* **`dev_base_url`**: [https://raw.githubusercontent.com/delveopers/biosaic/dev/vocab/](https://raw.githubusercontent.com/delveopers/biosaic/dev/vocab/)
* **`hugginface_url`**: [https://huggingface.co/shivendrra/BiosaicTokenizer/](https://huggingface.co/shivendrra/BiosaicTokenizer/)

URLs auto‑generate from `encoding` (e.g. `base_3k.model` or `cont_4k.model`).

## 5. Dependencies & Compatibility

* **Python:** 3.8 <= x
* **Dependencies:**

  * `requests>=2.25.1` (HTTP fetching)
* **Testing:**

  * `pytest` for unit tests
* **Packaging:**

  * `setuptools` & `wheel` via `pyproject.toml`

## 6. Performance & Limitations

* **Vocab Size Explosion:**

  * Continuous Protein & DNA k‑mers: `21ⁿ` & `5ⁿ` permutations.
  * Choose k ≤ 4–5 for tractable memory.
* **Edge Deployment:**

  * Downloads vocab once; can operate offline after initial load.
* **Thread Safety:**

  * Current classes are not explicitly thread‑safe due to temp‑file usage.
