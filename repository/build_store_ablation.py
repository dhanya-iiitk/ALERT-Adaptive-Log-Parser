"""
repository/build_store_ablation.py
====================================
Build template stores for component contribution ablation.
REUSES existing embeddings — no SBERT re-encoding needed!

Variants:
  --variant no_entropy    : fixed tau_d=0.5, no entropy adaptation
  --variant no_keyword    : single-stage clustering (no keyword partition)
  --variant no_prefix     : skip prefix restoration
  --variant freq_aware    : frequency-aware clustering

Usage:
  python3 repository/build_store_ablation.py --all --variant no_entropy
  python3 repository/build_store_ablation.py --all --variant no_keyword
  python3 repository/build_store_ablation.py --all --variant no_prefix
"""

import os, sys, re, json, hashlib, argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocess import DATASET_CONFIG
from repository.build_store import (
    DATASET_PARAMS, first_keyword,
    extract_templates, two_stage_cluster
)


def single_stage_cluster(embeddings, unique_logs, dist, dataset="", n_clusters=None):
    """No keyword partition — single-stage agglomerative clustering."""
    from sklearn.cluster import AgglomerativeClustering
    n = len(unique_logs)
    if n == 1:
        return np.zeros(1, dtype=int)
    if n_clusters and n_clusters > 0:
        return AgglomerativeClustering(
            n_clusters=n_clusters, metric="euclidean", linkage="average"
        ).fit_predict(embeddings)
    best_lbl = None
    best_score = -1e9
    for thr in [dist*0.7, dist*0.85, dist, dist*1.1]:
        thr = max(0.05, min(0.99, thr))
        try:
            lbl = AgglomerativeClustering(
                n_clusters=None, distance_threshold=thr,
                metric="euclidean", linkage="average"
            ).fit_predict(embeddings)
            sizes = list(Counter(lbl.tolist()).values())
            max_frac = max(sizes)/n
            if max_frac > 0.60: continue
            score = np.log1p(len(sizes)) - sum(1 for s in sizes if s==1)/len(sizes)*2 - max_frac*5
            if score > best_score:
                best_score = score
                best_lbl = lbl
        except: continue
    return best_lbl if best_lbl is not None else np.zeros(n, dtype=int)


def extract_templates_no_prefix(unique_logs, raw_contents, labels, tau_r):
    """Template extraction without prefix restoration."""
    cluster_logs = defaultdict(list)
    for i, lbl in enumerate(labels):
        cluster_logs[lbl].append(unique_logs[i])

    templates = {}
    for cid, logs in cluster_logs.items():
        if len(logs) == 1:
            templates[cid] = logs[0]
            continue
        max_len = max(len(l.split()) for l in logs)
        tokens_per_pos = [[] for _ in range(max_len)]
        for log in logs:
            toks = log.split()
            for pos, tok in enumerate(toks):
                tokens_per_pos[pos].append(tok)
        result = []
        for pos_toks in tokens_per_pos:
            if not pos_toks: continue
            mc = Counter(pos_toks).most_common(1)[0]
            result.append(mc[0] if mc[1]/len(logs) >= max(tau_r, 0.35) else "<*>")
        # Collapse wildcards only — NO prefix restoration
        collapsed = []
        for tok in result:
            if tok == "<*>" and collapsed and collapsed[-1] == "<*>": continue
            collapsed.append(tok)
        templates[cid] = " ".join(collapsed)
    return templates


def build_ablation_store(dataset, variant, data_dir="datasets",
                          store_dir="repository_ablation",
                          fixed_tau=0.5):
    print(f"\n{'='*60}")
    print(f"  Building [{variant}] store: {dataset}")
    print(f"{'='*60}")

    # Load existing embeddings — no SBERT needed!
    emb_path = os.path.join("repository", f"{dataset}_embeddings.npy")
    if not os.path.exists(emb_path):
        print(f"  ERROR: {emb_path} not found. Run build_store.py first.")
        return None

    embeddings = np.load(emb_path)
    print(f"  Loaded embeddings: {embeddings.shape}")

    # Load logs
    offline_csv = os.path.join(data_dir, dataset, f"{dataset}_offline.csv")
    online_csv  = os.path.join(data_dir, dataset, f"{dataset}_online.csv")
    df = pd.read_csv(offline_csv)
    if os.path.exists(online_csv):
        df = pd.concat([df, pd.read_csv(online_csv)], ignore_index=True)

    unique_logs  = df["normalised"].dropna().unique().tolist()
    # Support both column name formats
    raw_col = "raw_content" if "raw_content" in df.columns else "content"
    raw_contents = []
    for u in unique_logs:
        match = df[df["normalised"] == u][raw_col].iloc[0]
        raw_contents.append(match)

    dist, tau, n_clusters = DATASET_PARAMS.get(dataset, (0.70, 0.50, None))

    # Apply variant
    if variant == "no_entropy":
        # Use fixed tau_d=0.5 instead of per-dataset optimal
        labels = two_stage_cluster(
            embeddings, unique_logs, fixed_tau,
            dataset=dataset, n_clusters=n_clusters)

    elif variant == "no_keyword":
        # Single-stage clustering — no keyword partition
        labels = single_stage_cluster(
            embeddings, unique_logs, dist,
            dataset=dataset, n_clusters=n_clusters)

    elif variant == "no_prefix":
        # Normal clustering but skip prefix restoration
        labels = two_stage_cluster(
            embeddings, unique_logs, dist,
            dataset=dataset, n_clusters=n_clusters)

    else:
        print(f"Unknown variant: {variant}")
        return None

    # Template extraction
    if variant == "no_prefix":
        templates_map = extract_templates_no_prefix(
            unique_logs, raw_contents, labels, tau)
    else:
        templates_map = extract_templates(
            unique_logs, raw_contents, labels, tau)

    # Centroids
    centroids = {}
    for cid in set(labels):
        mask = labels == cid
        c    = embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[cid] = c / (norm + 1e-8)

    # Save store
    os.makedirs(store_dir, exist_ok=True)
    store_path = os.path.join(store_dir, f"{dataset}_template_store.json")
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for cid, template in templates_map.items():
        centroid = centroids.get(cid, np.zeros(embeddings.shape[1]))
        tid      = hashlib.md5(template.encode()).hexdigest()[:8]
        count    = int((labels == cid).sum())
        records.append({
            "template_id": tid,
            "template":    template,
            "centroid":    centroid.tolist(),
            "occurrences": count,
            "first_seen":  now,
            "last_seen":   now,
            "source":      "parser",
            "dataset":     dataset,
        })

    with open(store_path, "w") as f:
        json.dump(records, f, indent=2)

    np.save(os.path.join(store_dir, f"{dataset}_embeddings.npy"), embeddings)

    print(f"  [{dataset}] Templates={len(records)}  Store={store_path}")
    return {"dataset": dataset, "n_templates": len(records)}


def main():
    parser = argparse.ArgumentParser(
        description="Build ablation stores (reuses existing embeddings)")
    parser.add_argument("--dataset",   type=str, default=None)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--variant",   type=str, required=True,
        choices=["no_entropy","no_keyword","no_prefix"],
        help="Ablation variant to build")
    parser.add_argument("--data_dir",  type=str, default="datasets")
    parser.add_argument("--fixed_tau", type=float, default=0.5,
        help="Fixed tau_d for no_entropy variant")
    args = parser.parse_args()

    store_dir = f"repository_{args.variant}"
    datasets  = list(DATASET_PARAMS.keys()) if args.all else [args.dataset]

    for ds in datasets:
        if ds not in DATASET_PARAMS:
            print(f"Unknown dataset: {ds}"); continue
        build_ablation_store(
            ds, args.variant, args.data_dir, store_dir, args.fixed_tau)

    print(f"\nAll stores saved to: {store_dir}/")


if __name__ == "__main__":
    main()
