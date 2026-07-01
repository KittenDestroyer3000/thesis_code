# Penco Module EIA Analysis Pipeline

Computational pipeline supporting the thesis *"Rare earth mining in the Biobío region, Chile:
how well were public concerns addressed?"*, analysing the ICSARA/adenda dialogue for the Penco Module rare earth mining project (Aclara Resources, Biobío Region, Chile) across three rounds of environmental review.

## Overview

This repository contains the full data science pipeline used to:

1. Extract and preprocess text from adenda and ICSARA PDF documents
2. Translate Spanish source text to English
3. Model adenda topics using Latent Dirichlet Allocation (LDA)
4. Classify ICSARA concerns into adenda topics using LLM-based classification
5. Construct a master topic codebook to enable cross-round comparison
6. Score adenda responsiveness to individual ICSARA concerns using retrieval-augmented LLM assessment
7. Visualise thematic representation gaps and coverage scores across rounds

## Pipeline structure

### Adenda processing
| Script | Description |
|---|---|
| `addenda_txt.py` | PDF to text extraction (pdfplumber, EasyOCR fallback) |
| `preprocess.py` | Text to page-based chunks |
| `translate_openai.py` | Spanish to English translation |
| `lda_model.py` | LDA topic modelling |
| `visualise_topics.py` | Adenda topic visualisations |

### ICSARA processing
| Script | Description |
|---|---|
| `extract_icsara.py` | PDF to Spanish text extraction |
| `parse_icsara.py` | Structured parsing into individual concerns |
| `translate_icsara.py` | Spanish to English translation |
| `classify_icsara.py` | LLM classification into adenda topic codebook |
| `visualise_icsara.py` | ICSARA classification visualisations |

### Cross-round mapping and evaluation
| Script | Description |
|---|---|
| `master_codebook.csv` | Manually constructed mapping of round-specific topics to 9 master topics |
| `embed_adenda.py` | Sentence embedding of adenda chunks (all-mpnet-base-v2) |
| `coverage_icsara.py` | Retrieval-augmented LLM coverage scoring |
| `visualise_coverage.py` | Coverage score visualisations (by topic, by section) |
| `visualise_coverage_combined.py` | Combined coverage distribution across rounds |
| `visualise_crossround.py` | Cross-round representation gap and comparison charts |
| `evaluate_topics.py` | Topic coherence evaluation (C_v, NPMI, gensim) |

### Utilities
| Script | Description |
|---|---|
| `prepare_explorer.py` | Builds the interactive HTML explorer |
| `count_source_pdfs.py` | Counts and unzips source document archives |

## Interactive explorer

An interactive tool for browsing ICSARA items, adenda topics, and coverage scores is available at:
**https://kittendestroyer3000.github.io/interactive_explorer/**

## Data availability

Source PDF documents (adenda and ICSARA reports) are publicly available through the Chilean Environmental Impact Assessment System (SEA):
https://www.sea.gob.cl

Raw source documents, translated text, and intermediate pipeline outputs are not included in this repository due to file size. Processed outputs (topic distributions, classification results, coverage scores) required to reproduce the thesis figures and tables are available in `/data/`.

## Requirements

```
pip install -r requirements.txt
```

Key dependencies: `pandas`, `scikit-learn`, `sentence-transformers`, `gensim`, `matplotlib`, `openai`, `pdfplumber`, `easyocr`

## Reproducibility note

Translation, ICSARA classification, and coverage scoring steps use OpenAI's `gpt-4o-mini` model via API. As with any LLM-based process, exact outputs are not guaranteed to be identical across runs — see the Limitations section of the accompanying thesis for a full discussion.

## Citation

If referencing this work, please cite:

> Schulten, O (2026). *Rare earth mining in the Biobío region, Chile:
how well were public concerns addressed?*. Utrecht University.
