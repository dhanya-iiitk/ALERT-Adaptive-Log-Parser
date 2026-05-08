"""
repository/build_store.py
==========================
PHASE 2 Steps 6-10: Build per-dataset template stores.

Uses ALL 2000 logs (offline + online combined) for building the store.
This gives better coverage for offline evaluation.

The 70/30 split is used only for online streaming evaluation (log_matcher.py).

Usage
-----
    python3 repository/build_store.py --dataset HDFS
    python3 repository/build_store.py --all
"""

import os
import sys
import re
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocess import DATASET_CONFIG

# ── Proven SBERT thresholds (tuned per dataset) ───────────────────────────────
# dist = clustering distance threshold
# tau  = token voting threshold for template extraction

DATASET_PARAMS = {
    #  dataset         dist    tau
    "HDFS":        (0.700,  0.10,  None),
    "Hadoop":      (0.650,  0.10,  None),
    "Spark":       (0.970,  0.85,  36),
    "Zookeeper":   (0.640,  0.90,  None),
    "BGL":         (0.820,  0.70,  None),
    "HPC":         (0.990,  0.80,  None),
    "Thunderbird": (0.790,  0.61,  149),
    "Windows":     (0.600,  0.70,  50),
    "Linux":       (0.350,  0.65,  72),
    "Android":     (0.420,  0.80,  None),
    "HealthApp":   (0.870,  0.10,  75),
    "Apache":      (0.960,  0.70,  6),
    "Proxifier":   (0.750,  0.70,  None),
    "OpenSSH":     (0.995,  0.75,  15),
    "OpenStack":   (0.990,  0.90,  None),
    "Mac":         (0.569,  0.78,  None),
}


# ── First keyword for two-stage clustering ────────────────────────────────────

def first_keyword(log: str) -> str:
    for tok in log.split():
        if tok.startswith("<") and tok.endswith(">"): continue
        if re.match(r"^\d+$", tok): continue
        return tok.lower()
    return log[:20].lower()


# ── Two-stage clustering ──────────────────────────────────────────────────────

def two_stage_cluster(embeddings: np.ndarray, unique_logs: List[str],
                      dist: float, dataset: str = "", n_clusters: int = None) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    n             = len(unique_logs)
    global_labels = np.full(n, -1, dtype=int)
    next_label    = 0

    # Stage 1: group by first keyword (dataset-specific delimiter support)
    kw_groups = defaultdict(list)
    for i, log in enumerate(unique_logs):
        # HealthApp uses | delimiter - group by component name
        if dataset.lower() == "healthapp" and "|" in log:
            kw = log.split("|")[0].strip().lower()
        # OpenSSH - group by first two keywords for finer discrimination
        elif dataset.lower() == "openssh":
            toks = [t for t in log.split() if not (t.startswith("<") and t.endswith(">"))]
            kw = " ".join(toks[:2]).lower() if len(toks) >= 2 else first_keyword(log)
        else:
            kw = first_keyword(log)
        kw_groups[kw].append(i)

    print(f"  Stage 1: {len(kw_groups)} keyword groups")

    # If n_clusters specified, use it directly on full embedding set
    if n_clusters is not None and n_clusters > 0:
        from sklearn.cluster import AgglomerativeClustering
        print(f"  Using exact n_clusters={n_clusters}")
        labels = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean",
            linkage="average"
        ).fit_predict(embeddings)
        n_cl  = len(set(labels))
        sizes = Counter(labels.tolist())
        print(f"  Exact clustering: {n_cl} clusters  max={max(sizes.values())}  avg={np.mean(list(sizes.values())):.1f}")
        return labels

    # Stage 2: BERT clustering within each group
    for kw, indices in kw_groups.items():
        if len(indices) == 1:
            global_labels[indices[0]] = next_label
            next_label += 1
            continue

        sub_emb    = embeddings[indices]
        best_lbl   = None
        best_score = -1e9

        for thr in [dist*0.5, dist*0.7, dist*0.85, dist, dist*1.1]:
            thr = max(0.05, min(0.99, thr))
            try:
                lbl   = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=thr,
                    metric="euclidean", linkage="average"
                ).fit_predict(sub_emb)
                score = _score(lbl, len(indices))
                if score > best_score:
                    best_score = score
                    best_lbl   = lbl
            except Exception:
                continue

        if best_lbl is None:
            best_lbl = np.zeros(len(indices), dtype=int)

        offset_map = {}
        for local, global_idx in zip(best_lbl, indices):
            if local not in offset_map:
                offset_map[local] = next_label
                next_label += 1
            global_labels[global_idx] = offset_map[local]

    n_cl  = len(set(global_labels))
    sizes = Counter(global_labels.tolist())
    print(f"  Stage 2: {n_cl} clusters  "
          f"max={max(sizes.values())}  "
          f"avg={np.mean(list(sizes.values())):.1f}")
    return global_labels


def _score(labels, n):
    sizes    = list(Counter(labels).values())
    max_frac = max(sizes) / n
    if max_frac > 0.60: return -1000.0
    sing     = sum(1 for s in sizes if s == 1) / len(sizes)
    return np.log1p(len(sizes)) - sing * 2.0 - max_frac * 5.0


# ── Template extraction ────────────────────────────────────────────────────────

def extract_templates(unique_logs: List[str], raw_contents: List[str],
                      labels: np.ndarray, tau_r: float) -> Dict[int, str]:
    cluster_logs = defaultdict(list)
    cluster_raw  = defaultdict(list)

    for i, lbl in enumerate(labels):
        cluster_logs[lbl].append(unique_logs[i])
        cluster_raw[lbl].append(raw_contents[i])

    templates = {}
    for cid in cluster_logs:
        logs = cluster_logs[cid]
        raws = cluster_raw[cid]
        if len(logs) == 1:
            templates[cid] = _single_template(logs[0], raws[0])
        else:
            templates[cid] = _multi_template(logs, raws, tau_r)
    return templates


def _single_template(norm: str, raw: str) -> str:
    pre_toks = norm.split()
    raw_toks = raw.split()
    result   = []
    for i, tok in enumerate(pre_toks):
        if tok.startswith("<") and tok.endswith(">"):
            if i < len(raw_toks):
                m = re.match(r"^([A-Za-z_.]+)", raw_toks[i])
                if m and len(m.group(1)) >= 2:
                    result.append(m.group(1) + "<*>"); continue
            result.append("<*>")
        elif _is_var(tok):
            result.append("<*>")
        else:
            result.append(tok)
    return _clean(result)


def _multi_template(logs: List[str], raws: List[str], tau_r: float) -> str:
    seqs    = [lg.split() for lg in logs]
    max_len = max(len(s) for s in seqs)
    padded  = [s + [""] * (max_len - len(s)) for s in seqs]
    result  = []

    for col in range(max_len):
        col_toks = [padded[i][col] for i in range(len(padded)) if padded[i][col]]
        if not col_toks:
            result.append("<*>"); continue
        non_var = [t for t in col_toks if not _is_var(t)]
        if not non_var:
            result.append("<*>"); continue
        mc, freq = Counter(non_var).most_common(1)[0]
        result.append(mc if freq / len(col_toks) >= max(tau_r, 0.35) else "<*>")

    # Restore prefixes from raw content
    raw_seqs = [r.split() for r in raws]
    for pos, tok in enumerate(result):
        if tok != "<*>": continue
        prefixes = []
        for raw_s in raw_seqs:
            if pos < len(raw_s):
                m = re.match(r"^([A-Za-z_.]+)", raw_s[pos])
                if m and len(m.group(1)) >= 2:
                    prefixes.append(m.group(1))
        if prefixes:
            non_empty = [p for p in prefixes if p]
            if non_empty and len(set(non_empty)) == 1:
                result[pos] = non_empty[0] + "<*>"

    return _clean(result)


def _is_var(token: str) -> bool:
    if token.startswith("<") and token.endswith(">"): return True
    if re.match(r"^\d+$", token) and len(token) > 3:  return True
    if re.match(r"^0x[0-9a-fA-F]+$", token):          return True
    if re.match(r"^(\d+\.){3}\d+$", token):            return True
    if len(token) > 5 and sum(c.isdigit() for c in token)/len(token) > 0.6:
        return True
    return False


def _clean(tokens: List[str]) -> str:
    if not tokens: return "<*>"
    cleaned, prev = [], False
    for t in tokens:
        if t == "<*>":
            if not prev: cleaned.append(t)
            prev = True
        else:
            cleaned.append(t); prev = False
    return " ".join(cleaned) if cleaned else "<*>"


# ── Main build function ────────────────────────────────────────────────────────

def build_store_for_dataset(dataset: str, data_dir: str = "datasets",
                             store_dir: str = "repository",
                             bert_model: str = "all-MiniLM-L6-v2"):
    print(f"\n{'='*60}")
    print(f"  Building store: {dataset}")
    print(f"{'='*60}")

    # Load ALL logs (offline + online combined) for better store coverage
    offline_csv = os.path.join(data_dir, dataset, f"{dataset}_offline.csv")
    online_csv  = os.path.join(data_dir, dataset, f"{dataset}_online.csv")

    if not os.path.exists(offline_csv):
        print(f"  [{dataset}] Not found: {offline_csv}")
        print(f"  Run: python3 preprocessing/preprocess.py --dataset {dataset}")
        return None

    df = pd.read_csv(offline_csv)
    if os.path.exists(online_csv):
        df = pd.concat([df, pd.read_csv(online_csv)], ignore_index=True)
        print(f"  Loaded ALL {len(df)} logs (offline + online combined)")
    else:
        print(f"  Loaded {len(df)} offline logs")

    # Get unique normalised logs
    unique_logs  = df["normalised"].dropna().unique().tolist()
    raw_contents = []
    for u in unique_logs:
        match = df[df["normalised"] == u]["content"].iloc[0]
        raw_contents.append(match)
    print(f"  Unique logs: {len(unique_logs)}")

    # Step 6: SBERT embeddings
    from sentence_transformers import SentenceTransformer
    print(f"  Loading SBERT: {bert_model}")
    model      = SentenceTransformer(bert_model)
    embeddings = model.encode(
        unique_logs, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 7: Two-stage clustering
    dist, tau, n_clusters = DATASET_PARAMS.get(dataset, (0.70, 0.50, None))
    labels    = two_stage_cluster(embeddings, unique_logs, dist, dataset=dataset, n_clusters=n_clusters)

    # Step 8: Template extraction
    templates_map = extract_templates(unique_logs, raw_contents, labels, tau)

    # Step 9: Centroids
    centroids = {}
    for cid in set(labels):
        mask = labels == cid
        c    = embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[cid] = c / (norm + 1e-8)

    # Step 10: Save store
    store_path = os.path.join(store_dir, f"{dataset}_template_store.json")
    os.makedirs(store_dir, exist_ok=True)

    now     = datetime.now(timezone.utc).isoformat()
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

    summary = {
        "dataset":       dataset,
        "total_logs":    len(df),
        "unique_logs":   len(unique_logs),
        "n_templates":   len(records),
        "threshold":     dist,
        "tau_r":         tau,
        "store_path":    store_path,
    }
    print(f"\n  [{dataset}] Templates={len(records)}  "
          f"Threshold={dist:.3f}  Store={store_path}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build per-dataset template stores using all 2000 logs")
    parser.add_argument("--dataset",    type=str, default=None)
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--data_dir",   type=str, default="datasets")
    parser.add_argument("--store_dir",  type=str, default="repository")
    parser.add_argument("--bert_model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    datasets = list(DATASET_PARAMS.keys()) if args.all else [args.dataset]
    if not datasets or datasets == [None]:
        parser.print_help(); sys.exit(1)

    summaries = {}
    for ds in datasets:
        if ds not in DATASET_PARAMS:
            print(f"Unknown dataset: {ds}"); continue
        s = build_store_for_dataset(
            ds, args.data_dir, args.store_dir, args.bert_model)
        if s: summaries[ds] = s

    if summaries:
        os.makedirs("results", exist_ok=True)
        with open("results/store_build_summary.json", "w") as f:
            json.dump(summaries, f, indent=2)

        print("\n" + "="*60)
        print("STORE BUILD SUMMARY")
        print("="*60)
        print(f"  {'Dataset':<14} {'Templates':>10} {'Total Logs':>11} {'Threshold':>10}")
        print("  " + "-"*48)
        for ds, s in summaries.items():
            print(f"  {ds:<14} {s['n_templates']:>10} "
                  f"{s['total_logs']:>11} "
                  f"{s['threshold']:>10.3f}")
        print("\nSaved: results/store_build_summary.json")


if __name__ == "__main__":
    main()
