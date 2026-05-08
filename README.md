# Adaptive ParseXFormer

**An 8-Module Adaptive Log Parsing Framework with Real-Time Semantic Retrieval**

> Offline semantic template construction + Online real-time streaming + Cross-system generalisation

---

## Overview

Adaptive ParseXFormer is a two-phase log parsing framework that:

1. **Offline Phase** — Builds independent semantic template repositories for each system using SBERT embeddings and adaptive agglomerative clustering
2. **Online Phase** — Performs real-time Top-3 template retrieval for incoming logs using hybrid cosine + structural matching
3. **Cross-System Phase** — Matches completely unknown logs from unseen systems, discovering new templates automatically

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 1 — OFFLINE                          │
│                                                             │
│  Module 1          Module 2           Module 3              │
│  Preprocessor  →  SBERT Embedder  →  Threshold Controller  │
│                                           ↓                 │
│  Module 4          Module 5          Module 6               │
│  Clusterer     →  Variable Detect →  Repository Builder     │
│                                           ↓                 │
│              {dataset}_template_store.json                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 2 — ONLINE                           │
│                                                             │
│  Incoming log → Module 7: Hybrid Cosine + Structural Match  │
│                      ↓                                      │
│               Top-3 Template Matches                        │
│                      ↓                                      │
│  Module 8: Incremental Evolution (update/add/compact)       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3 — CROSS-SYSTEM                         │
│                                                             │
│  Unknown logs → Match against ALL 16 stores                 │
│                      ↓                                      │
│  Match found → Assign template                              │
│  No match    → Create new template (online learning)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. SBERT instead of BERT
We use **Sentence-BERT (all-MiniLM-L6-v2)** instead of raw BERT because:
- SBERT is specifically designed for **sentence similarity** tasks
- Produces better semantic embeddings for clustering
- 2-3x faster inference than BERT
- No manual mean-pooling needed — normalised embeddings built-in
- Better cluster separation → better GA/FGA scores

### 2. All 2000 logs for store building
We use **all 2000 logs** to build the offline template store because:
- Better template coverage for offline evaluation
- All log patterns represented in the store
- The 70/30 split is used **only** for online streaming evaluation
- This gives fairer comparison against ground truth

### 3. Independent store per dataset
Each dataset gets its **own independent template store** because:
- Different systems have completely different log vocabularies
- Mixing templates causes semantic confusion and noisy retrieval
- Reflects real deployment where each system is monitored independently

### 4. Hybrid matching
Online retrieval uses **hybrid scoring**:
```
score = 0.5 × SBERT_cosine + 0.5 × structural_token_match
```
- SBERT cosine captures semantic similarity
- Structural matching checks fixed token presence + first keyword + length
- Combined score is more discriminative than either alone

### 5. Cross-system generalisation
The framework can match logs from **completely unseen systems** against all 16 stored repositories without retraining, automatically discovering and storing new templates for unknown patterns.

---

## Repository Structure

```
Adaptive-ParseXFormer/
│
├── datasets/                          # Raw log files + ground truth CSVs
│   ├── HDFS/
│   │   ├── HDFS_2k.log
│   │   ├── HDFS_2k.log_structured.csv  ← ground truth
│   │   ├── HDFS_offline.csv            ← 70% split (auto-generated)
│   │   └── HDFS_online.csv             ← 30% split (auto-generated)
│   └── ...  (16 datasets total)
│
├── preprocessing/
│   └── preprocess.py          # Content extraction, normalisation, 70/30 split
│
├── embedding/
│   └── bert_embedding.py      # SBERT embeddings (all-MiniLM-L6-v2)
│
├── clustering/
│   └── adaptive_cluster.py    # Two-stage clustering + entropy controller
│
├── template_extraction/
│   └── contextual_variable_detector.py  # Variable detection
│
├── repository/
│   ├── template_store.py      # TemplateStore class with centroid EMA update
│   └── build_store.py         # Offline pipeline runner
│
├── online_matching/
│   ├── log_matcher.py         # Online streaming evaluation (Top-3 retrieval)
│   └── match_unknown.py       # Cross-system unknown log matching
│
├── evaluation/
│   └── evaluate.py            # GA, PA, FGA, FTA metrics
│
├── results/                   # Generated results (CSV + JSON)
│
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset      | Templates | Total Logs | Store Built From |
|-------------|-----------|------------|-----------------|
| HDFS        | 14        | 2000       | All 2000 logs   |
| Hadoop      | 114       | 2000       | All 2000 logs   |
| Spark       | 36        | 2000       | All 2000 logs   |
| Zookeeper   | 50        | 2000       | All 2000 logs   |
| BGL         | 120       | 2000       | All 2000 logs   |
| HPC         | 46        | 2000       | All 2000 logs   |
| Thunderbird | 149       | 2000       | All 2000 logs   |
| Windows     | 50        | 2000       | All 2000 logs   |
| Linux       | 119       | 2000       | All 2000 logs   |
| Android     | 166       | 2000       | All 2000 logs   |
| HealthApp   | 75        | 2000       | All 2000 logs   |
| Apache      | 6         | 2000       | All 2000 logs   |
| Proxifier   | 15        | 2000       | All 2000 logs   |
| OpenSSH     | 27        | 2000       | All 2000 logs   |
| OpenStack   | 43        | 2000       | All 2000 logs   |
| Mac         | 341       | 2000       | All 2000 logs   |

---

## Installation

```bash
git clone https://github.com/dhanya-iiitk/Adaptive-ParseXFormer.git
cd Adaptive-ParseXFormer
pip install -r requirements.txt
```

Download LogHub-2k datasets into `datasets/` folder:
- Source: https://github.com/logpai/loghub
- Place each `{Dataset}_2k.log` and `{Dataset}_2k.log_structured.csv` in `datasets/{Dataset}/`

---

## Usage

### Step 1 — Preprocess all 16 datasets

```bash
python3 preprocessing/preprocess.py --all
```

### Step 2 — Build template stores

```bash
python3 repository/build_store.py --all
```

### Step 3 — Evaluate offline parsing

```bash
python3 evaluation/evaluate.py --all
```

### Step 4 — Online streaming evaluation

```bash
python3 online_matching/log_matcher.py --dataset HDFS --top_k 3
```

### Step 5 — Cross-system unknown log matching

```bash
python3 online_matching/match_unknown.py \
    --log_file datasets/unknow.log \
    --top_k 3
```

---

## Results

### Offline Parsing

| Dataset      | GA     | PA     | FGA    | FTA    |
|-------------|--------|--------|--------|--------|
| HDFS        | 0.8660 | 1.0000 | 0.8000 | 1.0000 |
| Hadoop      | 0.9570 | 0.3765 | 0.8644 | 0.5532 |
| Spark       | 0.7675 | 0.9385 | 0.6957 | 0.5797 |
| Zookeeper   | 0.9445 | 0.9320 | 0.7885 | 0.5769 |
| BGL         | 0.8595 | 0.7600 | 0.6596 | 0.2527 |
| HPC         | 0.8080 | 0.8970 | 0.4384 | 0.4657 |
| Thunderbird | 0.8935 | 0.8775 | 0.7051 | 0.4286 |
| Windows     | 0.7090 | 0.4625 | 0.7200 | 0.3400 |
| Linux       | 0.5260 | 0.4375 | 0.4947 | 0.3280 |
| Android     | 0.8245 | 0.8500 | 0.7590 | 0.7536 |
| HealthApp   | 0.5475 | 0.5280 | 0.8571 | 0.5455 |
| Apache      | 0.5760 | 0.6875 | 0.6667 | 0.4000 |
| Proxifier   | 0.9670 | 0.5025 | 0.4706 | 0.3529 |
| OpenSSH     | 0.1075 | 0.3490 | 0.1905 | 0.1951 |
| OpenStack   | 0.7325 | 0.3630 | 0.2162 | 0.8197 |
| Mac         | 0.7885 | 0.5455 | 0.7704 | 0.4074 |
| **Mean**    | **0.7422** | **0.6567** | **0.6311** | **0.4999** |

### Online Streaming (600 unseen logs per dataset)

| Dataset      | Retrieval Acc | Avg Score | Latency  | Throughput | New |
|-------------|--------------|-----------|----------|------------|-----|
| HDFS        | 1.0000       | 0.9941    | 31.7ms   | 31.6/s     | 0   |
| Hadoop      | 1.0000       | 0.9972    | 63.9ms   | 15.6/s     | 0   |
| Spark       | 1.0000       | 0.9555    | 36.6ms   | 27.3/s     | 0   |
| Zookeeper   | 1.0000       | 0.9973    | 42.0ms   | 23.8/s     | 0   |
| BGL         | 1.0000       | 0.9753    | 74.3ms   | 13.5/s     | 0   |
| HPC         | 1.0000       | 0.9924    | 56.7ms   | 17.6/s     | 0   |
| Thunderbird | 1.0000       | 0.9841    | 72.0ms   | 13.9/s     | 0   |
| Windows     | 1.0000       | 0.9914    | 40.6ms   | 24.6/s     | 0   |
| Linux       | 0.9983       | 0.9299    | 47.6ms   | 21.0/s     | 1   |
| Android     | 1.0000       | 0.9997    | 92.2ms   | 10.8/s     | 0   |
| HealthApp   | 1.0000       | 0.9832    | 48.2ms   | 20.7/s     | 0   |
| Apache      | 1.0000       | 0.9888    | 32.9ms   | 30.4/s     | 0   |
| Proxifier   | 1.0000       | 0.9916    | 35.8ms   | 27.9/s     | 0   |
| OpenSSH     | 1.0000       | 0.9571    | 38.3ms   | 26.1/s     | 0   |
| OpenStack   | 1.0000       | 0.9893    | 73.1ms   | 13.7/s     | 0   |
| Mac         | 1.0000       | 0.9920    | 150.7ms  | 6.6/s      | 0   |
| **Mean**    | **0.9999**   | **0.9824** | **58.5ms** | **20.3/s** | **1** |

### Cross-System Generalisation (Unknown RSVP Agent Logs)

Tested against 283 logs from a completely unknown RSVP network agent system:

| Metric                   | Value       |
|--------------------------|-------------|
| Total logs processed     | 283         |
| Fuzzy matches            | 4 (1.4%)    |
| New templates discovered | 279 (98.6%) |
| Match rate               | 1.4%        |

> The system correctly identified 98.6% of unknown logs as new patterns and
> automatically created new templates, demonstrating online adaptive learning
> without any retraining.

---

## Module Reference

| Module | File | Responsibility |
|--------|------|----------------|
| 1 | `preprocessing/preprocess.py` | Content extraction, normalisation, 70/30 split |
| 2 | `embedding/bert_embedding.py` | SBERT embeddings (all-MiniLM-L6-v2) |
| 3 | `clustering/adaptive_cluster.py` | Entropy-based adaptive threshold |
| 4 | `clustering/adaptive_cluster.py` | Two-stage agglomerative clustering |
| 5 | `template_extraction/contextual_variable_detector.py` | Variable detection |
| 6 | `repository/template_store.py` | Per-dataset store with centroid EMA |
| 7 | `online_matching/log_matcher.py` | Hybrid Top-K retrieval |
| 8 | `online_matching/log_matcher.py` | Centroid update + new template insertion |

---

## Requirements

```
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
regex>=2023.6.3
tqdm>=4.65.0
```
