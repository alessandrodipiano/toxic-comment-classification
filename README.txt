# ICTC – Internet Comments Toxicity Classification

> Text Mining and Natural Language Processing Project — A.Y. 2024/2025  
> Alessandro Di Piano (535269) · Malgorzata Ozarowska (535414)

---

## Overview

This project builds a multi-label text classification system to detect and categorize toxic behavior in online comments. A comment is considered toxic if it is rude, disrespectful, or otherwise likely to discourage participation in a discussion.

The model classifies each comment into one or more of the following categories:

| Label | Description |
|---|---|
| `toxic` | General toxicity |
| `severe_toxic` | Extreme toxicity |
| `obscene` | Obscene language |
| `threat` | Threatening language |
| `insult` | Insulting language |
| `identity_hate` | Hate speech targeting identity |

---

## Dataset

- **Source:** Wikipedia editors' discussion forum (via [Kaggle](https://www.kaggle.com/))
- **Size:** 159,570 comments
- **Language:** English (heavy use of internet slang)
- **Class balance:** Highly imbalanced — ~89.8% of comments are neutral

**Test set label distribution:**

| Label | Count | % |
|---|---|---|
| toxic | 1530 | 9.59% |
| severe_toxic | 160 | 1.0% |
| obscene | 845 | 5.3% |
| threat | 48 | 0.3% |
| insult | 788 | 4.94% |
| identity_hate | 141 | 0.88% |

**Comment length statistics:**

|  | Words | Subwords |
|---|---|---|
| Longest | 1235 | 1807 |
| Shortest | 0 | 2 |
| Average | 32.1 | 79.1 |
| Median | 15 | 39 |

### Notable Data Challenges

- Heavy internet slang leads to a very large unique vocabulary
- ASCII art and near-unreadable comments pose challenges even for advanced models
- The **SolidGoldMagicarp phenomenon**: usernames and repeated entities (e.g. film/game titles) create noise
- Word frequency follows **Zipf's law**; top words after stopword removal are either Wikipedia-specific terms or slurs

---

## Methodology

The project explores the full NLP pipeline, testing multiple choices at each stage:

### 1. Preprocessing
- **Basic:** strip non-alphabetical characters, remove stopwords
- **Uncleaned:** retain raw text so models can infer from formatting cues

### 2. Tokenization
- **Word tokenizer** (NLTK) — applied to basic-preprocessed text
- **Subword tokenizer** (WordPiece, HuggingFace `tokenizers`) — applied to uncleaned text

### 3. Embeddings
- **PPMI** — positive pointwise mutual information (term-term co-occurrence matrix)
- **TF-IDF** — term-document frequency weighting
- **Word2Vec** — pretrained gensim model (window size 5)
- **Task-specific embeddings** — learned directly via a neural `Embedding` layer (subword-based)

SVD (Latent Semantic Analysis) was applied to both PPMI and TF-IDF matrices to reduce sparsity and extract latent semantic dimensions.

### 4. Models

| Model | Embeddings | Notes |
|---|---|---|
| Logistic Regression | Word2Vec or PPMI | Ensemble of 6 binary classifiers (one per label) |
| RNN + LSTM | Word2Vec / PPMI / TF-IDF / trainable | 4 variants; same architecture |
| BERT | Transformer (fine-tuned) | `bert-base-uncased` via HuggingFace Transformers |

**Training details:**
- Loss: Binary Cross-Entropy (sigmoid outputs, one per label)
- Class imbalance handled via **inverse-frequency instance weighting** (~+0.05 F1 improvement)
- Optimizer: Adam (RNN), AdamW (BERT)
- BERT uses `BCEWithLogitsLoss` (sigmoid applied internally)
- EarlyStopping used to prevent overfitting

---

## Results

Evaluation metrics:
- **F1 macro** — treats all classes equally regardless of frequency (suitable for imbalanced data)
- **ROC AUC** — measures ranking quality (metric recommended in the original task)

| Model | Threshold | F1 Macro | ROC AUC |
|---|---|---|---|
| RNN (subword) | 0.7 | 0.52 | 0.93 |
| RNN (Word2Vec) | 0.7 | 0.49 | 0.95 |
| RNN (PPMI) | 0.7 | 0.50 | 0.95 |
| RNN (TF-IDF) | 0.7 | 0.50 | 0.93 |
| **BERT** | **0.4** | **0.67** | **0.99** |

### Key Findings

1. **Logistic Regression** does not benefit from richer embeddings — aggregating a comment into one vector hides internal semantic relationships. PPMI-based LR required ~10× more iterations to converge.
2. **All RNN variants plateau** at similar performance levels, suggesting each model reaches the limit of what its embedding type can express with this architecture.
3. **BERT is the best model** overall, benefiting from pretraining, attention-based context modeling, and subword tokenization. Its superiority is visible in both aggregate metrics and per-comment inference.

### Bias Analysis

Models were tested on neutral-sounding sentences containing words that appear frequently in toxic contexts:

- **"Freddy Mercury was a gay man"** — All RNN models incorrectly flag this as toxic/identity-hate. BERT does not classify it as identity-hate or insult.
- **"stupid european"** — Correctly flagged as toxic by all models. TF-IDF-based RNN fails on identity-hate due to less semantically rich SVD dimensions.
- **"the monkey likes books"** — RNN models with static embeddings falsely flag this as toxic. BERT and the subword RNN correctly handle the neutral context.

BERT's attention mechanism allows it to model full context rather than relying on individual word associations, making it substantially more robust to bias.

---

## Project Structure

```
ICTC.ipynb          # Main notebook (all experiments)
Report_ICTC.pdf     # Full project report
```

---

## Dependencies

```
numpy
pandas
scikit-learn
scipy
nltk
gensim
tokenizers
tensorflow / keras
torch
transformers
matplotlib
seaborn
```

---

## Division of Work

**Malgorzata Ozarowska:**
- Text and data preprocessing
- Word frequency analysis
- Word2Vec embeddings
- Logistic Regression models
- RNN models

**Alessandro Di Piano:**
- PPMI and TF-IDF matrices, SVD
- T-SNE visualizations
- Latent Semantic Analysis
- BERT model and tokenizer
- Model prediction comparisons and bias evaluation











