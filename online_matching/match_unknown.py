"""
online_matching/match_unknown.py
=================================
Real-time matching of completely unknown logs against existing template stores.

Tries to match each log against ALL 16 dataset stores and returns
the best matching templates across all systems.

Usage
-----
    python3 online_matching/match_unknown.py --log_file datasets/unknow.log
    python3 online_matching/match_unknown.py --log_file datasets/unknow.log --top_k 3
    python3 online_matching/match_unknown.py --log_file datasets/unknow.log --dataset HDFS
"""

import os
import sys
import re
import json
import time
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["HDFS","Hadoop","Spark","Zookeeper","BGL","HPC","Thunderbird",
            "Windows","Linux","Android","HealthApp","Apache","Proxifier",
            "OpenSSH","OpenStack","Mac"]


def load_store(store_path: str) -> List[Dict]:
    with open(store_path) as f:
        data = json.load(f)
    for e in data:
        e["_centroid"] = np.array(e["centroid"], dtype=np.float32)
        toks = e["template"].split()
        e["_fixed"]    = [t for t in toks if t != "<*>"]
        e["_first_kw"] = toks[0].lower() if toks else ""
    return data


def preprocess(log: str) -> str:
    """Basic preprocessing for unknown logs."""
    # Mask IPs
    log = re.sub(r"(\d{1,3}\.){3}\d{1,3}(:\d+)?", "<IP>", log)
    # Mask numbers
    log = re.sub(r"\b\d+\b", "<NUM>", log)
    # Mask hex
    log = re.sub(r"\b0x[0-9a-fA-F]+\b", "<HEX>", log)
    # Normalise whitespace
    log = re.sub(r"\s+", " ", log).strip()
    return log


def structural_score(log_toks: List[str], entry: Dict) -> float:
    fixed   = entry["_fixed"]
    if not fixed: return 0.3
    log_set = set(t.lower() for t in log_toks)
    matches = sum(1 for t in fixed if t.lower() in log_set)
    token_score = matches / len(fixed)
    first_kw = 1.0 if (log_toks and
                        log_toks[0].lower() == entry["_first_kw"]) else 0.0
    tmpl_len = len(entry["template"].split())
    log_len  = len(log_toks)
    len_sim  = 1.0 - abs(log_len - tmpl_len) / max(log_len, tmpl_len, 1)
    return 0.5 * token_score + 0.3 * first_kw + 0.2 * len_sim


def match_against_store(
    vector:   np.ndarray,
    norm_log: str,
    store:    List[Dict],
    alpha:    float = 0.5,
    top_k:    int   = 3,
) -> List[Tuple[Dict, float]]:
    if not store: return []
    centroids = np.stack([e["_centroid"] for e in store])
    cos_scores = centroids @ vector
    log_toks   = norm_log.split()
    struct     = np.array([structural_score(log_toks, e) for e in store])
    hybrid     = alpha * cos_scores + (1 - alpha) * struct
    top_idx    = np.argsort(hybrid)[::-1][:top_k]
    return [(store[i], float(hybrid[i])) for i in top_idx]


def extract_content(log: str) -> str:
    """Extract meaningful content from unknown log format."""
    # Try to strip common header patterns: date time level component: content
    patterns = [
        r"^\S+\s+\S+\s+\w+\s+[:\w\.]+:\s*(.*)",   # date time level component: content
        r"^\S+\s+\S+\s+\w+\s+(.*)",                 # date time level content
        r"^\w+/\d+\s+\S+\s+\w+\s+[:\.\w]+\s+(.*)", # month/day time level component content
    ]
    for pat in patterns:
        m = re.match(pat, log.strip())
        if m:
            return m.group(1).strip()
    return log.strip()


def run_unknown_matching(
    log_file:   str,
    store_dir:  str   = "repository",
    dataset:    Optional[str] = None,
    top_k:      int   = 3,
    model_name: str   = "all-MiniLM-L6-v2",
    new_thr:    float = 0.60,
    output_dir: str   = "results",
):
    # Load logs
    with open(log_file, encoding="utf-8", errors="ignore") as f:
        raw_logs = [l.strip() for l in f if l.strip()]
    print(f"\nLoaded {len(raw_logs)} unknown logs from {log_file}")

    # Load stores
    if dataset:
        datasets_to_search = [dataset]
    else:
        datasets_to_search = DATASETS

    stores = {}
    for ds in datasets_to_search:
        path = os.path.join(store_dir, f"{ds}_template_store.json")
        if os.path.exists(path):
            stores[ds] = load_store(path)
            print(f"  Loaded {ds} store: {len(stores[ds])} templates")

    if not stores:
        print("No stores found! Run build_store.py first.")
        return

    # Load SBERT
    from sentence_transformers import SentenceTransformer
    print(f"\nLoading SBERT: {model_name}")
    model = SentenceTransformer(model_name)

    # Process each log
    results       = []
    new_templates = []
    match_counts  = {"exact": 0, "partial": 0, "fuzzy": 0, "new": 0}

    print(f"\n{'='*70}")
    print(f"REAL-TIME MATCHING — {len(raw_logs)} UNKNOWN LOGS")
    print(f"{'='*70}\n")

    for i, raw_log in enumerate(raw_logs):
        t0      = time.perf_counter()
        content = extract_content(raw_log)
        norm    = preprocess(content)
        vector  = model.encode([norm], convert_to_numpy=True,
                                normalize_embeddings=True)[0].astype(np.float32)

        # Search all stores
        all_matches = []
        for ds, store in stores.items():
            matches = match_against_store(vector, norm, store,
                                          alpha=0.5, top_k=top_k)
            for entry, score in matches:
                all_matches.append((ds, entry, score))

        # Sort by score
        all_matches.sort(key=lambda x: x[2], reverse=True)
        top_matches = all_matches[:top_k]

        latency = (time.perf_counter() - t0) * 1000

        if not top_matches or top_matches[0][2] < new_thr:
            # New unseen template
            match_counts["new"] += 1
            new_templates.append(norm)
            match_type = "new"
            print(f"Log {i+1:3d}: [NEW    ] {content[:60]}")
            print(f"         Template: {norm[:60]}")
        else:
            best_ds, best_entry, best_score = top_matches[0]
            if best_score >= 0.99:
                match_type = "exact"; match_counts["exact"] += 1
            elif best_score >= 0.80:
                match_type = "partial"; match_counts["partial"] += 1
            else:
                match_type = "fuzzy"; match_counts["fuzzy"] += 1

            print(f"Log {i+1:3d}: [{match_type.upper():7s}] {content[:55]}")
            for rank, (ds, entry, score) in enumerate(top_matches[:top_k], 1):
                print(f"  Rank {rank} [{ds:12s}] score={score:.4f} | {entry['template'][:55]}")

        print(f"  Latency: {latency:.1f}ms")
        print()

        results.append({
            "log_id":       i + 1,
            "raw_log":      raw_log[:100],
            "content":      content[:100],
            "normalised":   norm[:100],
            "match_type":   match_type,
            "top1_dataset": top_matches[0][0] if top_matches and match_type != "new" else "NEW",
            "top1_template":top_matches[0][1]["template"] if top_matches and match_type != "new" else norm,
            "top1_score":   round(top_matches[0][2], 4) if top_matches and match_type != "new" else 0.0,
        })

    # Summary
    total = len(raw_logs)
    print(f"\n{'='*70}")
    print(f"REAL-TIME MATCHING SUMMARY")
    print(f"{'='*70}")
    print(f"  Total logs processed : {total}")
    print(f"  Exact matches        : {match_counts['exact']}  ({match_counts['exact']/total*100:.1f}%)")
    print(f"  Partial matches      : {match_counts['partial']}  ({match_counts['partial']/total*100:.1f}%)")
    print(f"  Fuzzy matches        : {match_counts['fuzzy']}  ({match_counts['fuzzy']/total*100:.1f}%)")
    print(f"  New templates found  : {match_counts['new']}  ({match_counts['new']/total*100:.1f}%)")
    print(f"  Match rate           : {(total-match_counts['new'])/total*100:.1f}%")

    if new_templates:
        print(f"\n  New templates discovered:")
        for t in new_templates[:10]:
            print(f"    - {t[:65]}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "unknown_log_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "log_file":    log_file,
            "total_logs":  total,
            "match_counts":match_counts,
            "results":     results,
            "new_templates": new_templates,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time matching of unknown logs against template stores")
    parser.add_argument("--log_file",   type=str, required=True,
                        help="Path to unknown log file")
    parser.add_argument("--dataset",    type=str, default=None,
                        help="Specific dataset store to search (default: all)")
    parser.add_argument("--store_dir",  type=str, default="repository")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--top_k",      type=int, default=3)
    parser.add_argument("--new_thr",    type=float, default=0.60)
    parser.add_argument("--model",      type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    run_unknown_matching(
        log_file   = args.log_file,
        store_dir  = args.store_dir,
        dataset    = args.dataset,
        top_k      = args.top_k,
        model_name = args.model,
        new_thr    = args.new_thr,
        output_dir = args.output_dir,
    )


if __name__ == "__main__":
    main()
