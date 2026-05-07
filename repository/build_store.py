"""
repository/build_store.py
==========================
PHASE 2 Steps 6-10: For each dataset independently:
  - Load offline split
  - Generate BERT embeddings
  - Adaptive clustering
  - Template extraction
  - Build dataset-specific template store

Usage
-----
    python repository/build_store.py --dataset HDFS
    python repository/build_store.py --all
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import DATASET_CONFIG, SPLIT_RATIO


DATASET_PARAMS = {
    #  dataset       dist    tau
    "HDFS":        (0.250,  0.10),
    "Hadoop":      (0.090,  0.10),
    "Spark":       (0.160,  0.85),
    "Zookeeper":   (0.140,  0.90),
    "BGL":         (0.090,  0.70),
    "HPC":         (0.140,  0.80),
    "Thunderbird": (0.080,  0.61),
    "Windows":     (0.140,  0.70),
    "Linux":       (0.090,  0.65),
    "Android":     (0.080,  0.80),
    "HealthApp":   (0.110,  0.10),
    "Apache":      (0.250,  0.70),
    "Proxifier":   (0.240,  0.70),
    "OpenSSH":     (0.180,  0.75),
    "OpenStack":   (0.140,  0.90),
    "Mac":         (0.060,  0.78),
}


def build_store_for_dataset(
    dataset:   str,
    data_dir:  str = "datasets",
    store_dir: str = "repository",
    bert_model: str = "bert-base-uncased",
):
    """
    Steps 6–10 for one dataset.
    Reads {dataset}_offline.csv, runs full offline pipeline,
    saves {dataset}_template_store.json.
    """
    print(f"\n{'='*60}")
    print(f"  Building store: {dataset}")
    print(f"{'='*60}")

    # ── Load offline split ─────────────────────────────────────────
    offline_csv = os.path.join(data_dir, dataset, f"{dataset}_offline.csv")
    if not os.path.exists(offline_csv):
        print(f"  [{dataset}] Offline CSV not found: {offline_csv}")
        print(f"  Run: python preprocessing/preprocess.py --dataset {dataset}")
        return None

    df = pd.read_csv(offline_csv)
    print(f"  Loaded {len(df)} offline logs")

    # Step 1.3: get unique normalised logs
    unique_logs  = df["normalised"].dropna().unique().tolist()
    raw_contents = []
    for u in unique_logs:
        match = df[df["normalised"] == u]["content"].iloc[0]
        raw_contents.append(match)
    print(f"  Unique logs: {len(unique_logs)}")

    # ── Step 6: BERT embeddings ────────────────────────────────────
    from embedding.bert_embedding import generate_embeddings
    embeddings, attentions = generate_embeddings(
        unique_logs, bert_model=bert_model, return_attentions=True)

    # ── Step 7: Adaptive clustering ────────────────────────────────
    dist, tau = DATASET_PARAMS.get(dataset, (0.20, 0.50))
    from clustering.adaptive_cluster import cluster_logs
    labels = cluster_logs(embeddings, distance_threshold=dist)

    # ── Step 8: Template extraction ────────────────────────────────
    from template_extraction.contextual_variable_detector import extract_all_templates
    templates_map = extract_all_templates(
        unique_logs, raw_contents, labels, attentions,
        tokenizer_name=bert_model, tau_r=tau)

    # ── Step 9: Compute centroids ──────────────────────────────────
    centroids = _compute_centroids(embeddings, labels)

    # ── Step 10: Build and save repository ────────────────────────
    store_path = os.path.join(store_dir, f"{dataset}_template_store.json")
    os.makedirs(store_dir, exist_ok=True)

    records = []
    import hashlib
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    for cid, template in templates_map.items():
        centroid = centroids.get(cid, np.zeros(768))
        tid = hashlib.md5(template.encode()).hexdigest()[:8]
        count = int((labels == cid).sum())
        records.append({
            "template_id":  tid,
            "template":     template,
            "centroid":     centroid.tolist(),
            "occurrences":  count,
            "first_seen":   now,
            "last_seen":    now,
            "source":       "parser",
            "dataset":      dataset,
        })

    with open(store_path, "w") as f:
        json.dump(records, f, indent=2)

    # Save embeddings for potential reuse
    emb_path = os.path.join(store_dir, f"{dataset}_embeddings.npy")
    np.save(emb_path, embeddings)

    summary = {
        "dataset":       dataset,
        "total_offline": len(df),
        "unique_logs":   len(unique_logs),
        "n_templates":   len(records),
        "store_path":    store_path,
        "dist":          dist,
        "tau":           tau,
    }
    print(f"\n  [{dataset}] Templates: {len(records)}  Store: {store_path}")
    return summary


def _compute_centroids(embeddings, labels):
    centroids = {}
    for cid in set(labels):
        mask = labels == cid
        c = embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[cid] = c / (norm + 1e-8)
    return centroids


def main():
    parser = argparse.ArgumentParser(description="Build per-dataset template stores")
    parser.add_argument("--dataset",    type=str,  default=None)
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--data_dir",   type=str,  default="datasets")
    parser.add_argument("--store_dir",  type=str,  default="repository")
    parser.add_argument("--bert_model", type=str,  default="bert-base-uncased")
    args = parser.parse_args()

    datasets = list(DATASET_PARAMS.keys()) if args.all else [args.dataset]
    if not datasets or datasets == [None]:
        parser.print_help(); sys.exit(1)

    summaries = {}
    for ds in datasets:
        summary = build_store_for_dataset(
            ds, args.data_dir, args.store_dir, args.bert_model)
        if summary:
            summaries[ds] = summary

    if summaries:
        out = "results/store_build_summary.json"
        with open(out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
