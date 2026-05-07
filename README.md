# Adaptive ParseXFormer

**An 8-Module Adaptive Log Parsing Framework with Real-Time Semantic Retrieval**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 1 — OFFLINE                          │
│                                                             │
│  Module 1        Module 2        Module 3                   │
│  Preprocessor  → BERT Embedder → Threshold Controller       │
│                                       ↓                     │
│  Module 4        Module 5        Module 6                   │
│  Clusterer     → Variable Detect → Repository Builder       │
│                                       ↓                     │
│              {dataset}_template_store.json                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 2 — ONLINE                           │
│                                                             │
│  Incoming log → Module 7: Cosine Similarity Retrieval       │
│                      ↓                                      │
│               Top-3 Template Matches                        │
│                      ↓                                      │
│  Module 8: Incremental Evolution (update / add / compact)   │
└─────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
Adaptive-ParseXFormer/
│
├── datasets/                          # Raw log files (one folder per dataset)
│   ├── HDFS/
│   │   ├── HDFS_2k.log               # Raw logs
│   │   ├── HDFS_offline.csv          # 70% split (auto-generated)
│   │   └── HDFS_online.csv           # 30% split (auto-generated)
│   ├── BGL/
│   └── ...  (16 datasets total)
│
├── preprocessing/
│   └── preprocess.py                 # Steps 1-5: extract, normalise, split
│
├── embedding/
│   └── embedder.py                   # Step 6: BERT mean-pool embeddings
│
├── clustering/
│   └── clusterer.py                  # Step 7: Agglomerative + entropy control
│
├── template_extraction/
│   └── extractor.py                  # Step 8: Attention-variance templates
│
├── repository/
│   ├── build_store.py                # Step 10: Build per-dataset stores
│   ├── HDFS_template_store.json      # (auto-generated)
│   └── BGL_template_store.json       # (auto-generated)
│
├── online_matching/
│   └── stream_eval.py                # Steps 11-15: Streaming evaluation
│
├── evaluation/
│   └── evaluate_offline.py           # Step 9: GA, PA, FGA, FTA metrics
│
├── results/                          # All output CSVs and JSONs
│
├── notebooks/                        # Jupyter notebooks for analysis
│
├── requirements.txt
└── README.md
```

## Supported Datasets

| Dataset      | Templates | Offline logs | Online logs |
|-------------|-----------|-------------|-------------|
| HDFS        | 14        | 1,400       | 600         |
| Hadoop      | 114       | 1,400       | 600         |
| Spark       | 36        | 1,400       | 600         |
| Zookeeper   | 50        | 1,400       | 600         |
| BGL         | 120       | 1,400       | 600         |
| HPC         | 46        | 1,400       | 600         |
| Thunderbird | 149       | 1,400       | 600         |
| Windows     | 50        | 1,400       | 600         |
| Linux       | 119       | 1,400       | 600         |
| Android     | 166       | 1,400       | 600         |
| HealthApp   | 75        | 1,400       | 600         |
| Apache      | 6         | 1,400       | 600         |
| Proxifier   | 15        | 1,400       | 600         |
| OpenSSH     | 27        | 1,400       | 600         |
| OpenStack   | 43        | 1,400       | 600         |
| Mac         | 341       | 1,400       | 600         |

> Each dataset has its **own independent template store**. Templates are never mixed across systems.

---

## Installation

```bash
git clone https://github.com/yourusername/Adaptive-ParseXFormer.git
cd Adaptive-ParseXFormer
pip install -r requirements.txt
```

Download LogHub datasets into `datasets/`:
```
datasets/HDFS/HDFS_2k.log
datasets/BGL/BGL_2k.log
... etc.
```

---

## Step-by-Step Usage

### Step 1 — Preprocess all datasets (70/30 split)

```bash
# Single dataset
python preprocessing/preprocess.py --dataset HDFS

# All 16 datasets
python preprocessing/preprocess.py --all
```

Output: `datasets/HDFS/HDFS_offline.csv` and `datasets/HDFS/HDFS_online.csv`

---

### Step 2 — Build template stores (offline phase)

```bash
# Single dataset
python repository/build_store.py --dataset HDFS

# All datasets
python repository/build_store.py --all
```

Output: `repository/HDFS_template_store.json`

---

### Step 3 — Evaluate offline parsing

```bash
python evaluation/evaluate_offline.py --all
```

Output: `results/offline_metrics.csv` with GA, PA, FGA, FTA for all 16 datasets.

---

### Step 4 — Online streaming evaluation

```bash
# Stream HDFS unseen logs, retrieve top-3 templates
python online_matching/stream_eval.py --dataset HDFS --top_k 3

# BGL streaming
python online_matching/stream_eval.py --dataset BGL --top_k 3
```

Output: `results/HDFS_stream_results.csv` and `results/HDFS_stream_summary.json`

---

## Online Metrics (Step 15)

| Metric                    | Description                                      |
|--------------------------|--------------------------------------------------|
| Retrieval accuracy        | Fraction of logs matched to existing template    |
| Exact match rate          | Cosine similarity ≥ 0.99                        |
| Partial match rate        | Cosine similarity ≥ 0.80                        |
| Fuzzy match rate          | Cosine similarity ≥ 0.60                        |
| New template discovery    | Fraction of logs creating new templates          |
| Avg / P95 latency (ms)   | Per-log processing time                          |
| Throughput (logs/sec)     | Real-time processing rate                        |
| Repository growth         | Final store size after streaming                 |

---

## Module Reference

| Module | File | Responsibility |
|--------|------|----------------|
| 1 | `preprocessing/preprocess.py` | Extract content, normalise, deduplicate, split 70/30 |
| 2 | `embedding/embedder.py` | BERT mean-pool embeddings + attention weights |
| 3 | `clustering/clusterer.py` | Entropy-based adaptive threshold (sliding window) |
| 4 | `clustering/clusterer.py` | Agglomerative clustering with best-threshold search |
| 5 | `template_extraction/extractor.py` | Attention-variance variable detection |
| 6 | `repository/build_store.py` | Template store with centroid vectors |
| 7 | `online_matching/stream_eval.py` | Cosine similarity top-K retrieval |
| 8 | `online_matching/stream_eval.py` | Centroid EMA update + new template insertion + compaction |

---

## Key Design Decisions

**Independent stores per dataset** — HDFS templates are never mixed with BGL templates. Each system gets its own `{dataset}_template_store.json`.

**70/30 temporal split** — Offline training uses the first 70% of logs; online streaming uses the remaining 30% in order, simulating real deployment.

**Centroid-based retrieval** — Cosine similarity between BERT embeddings avoids brittle string matching, handling log variants and paraphrases.

**Entropy-adaptive threshold** — Module 3 monitors cluster distribution entropy in a sliding window and adjusts τ_d dynamically: relaxes when system is diverse, tightens when repetitive.

**Attention-variance templates** — Module 5 uses BERT attention variance across cluster logs to identify variable positions, rather than pure frequency heuristics.
