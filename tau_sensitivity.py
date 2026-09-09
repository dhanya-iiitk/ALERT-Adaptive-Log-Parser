"""
tau_sensitivity.py

Empirically sweeps tau_r (token support threshold) and initial tau_d
(clustering distance threshold) for a chosen dataset, reusing the
EXACT production functions from repository/build_store.py
(two_stage_cluster, extract_templates), and evaluates each variant
against ground truth using the trusted metric functions from
evaluation/evaluate.py (compute_GA, compute_PA, compute_FGA, compute_FTA).

Efficiency: SBERT embeddings are computed ONCE per dataset (cached to
disk) since neither tau_r nor tau_d affects embedding generation.
  - Sweeping tau_d requires re-clustering (re-running two_stage_cluster)
    but NOT re-embedding.
  - Sweeping tau_r requires only re-running extract_templates on the
    SAME fixed clustering (the dataset's tuned tau_d) - no
    re-clustering needed at all.

Place in project root. Usage:
    python tau_sensitivity.py --dataset HDFS
    python tau_sensitivity.py --dataset Linux --sweep tau_r
    python tau_sensitivity.py --dataset Linux --sweep tau_d
    python tau_sensitivity.py --dataset Linux --sweep both
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repository.build_store import DATASET_PARAMS, two_stage_cluster, extract_templates  # noqa: E402
from evaluation.evaluate import compute_GA, compute_PA, compute_FGA, compute_FTA  # noqa: E402

TAU_R_VALUES = [0.10, 0.30, 0.50, 0.70, 0.90]
TAU_D_DELTA = [-0.20, -0.10, 0.00, 0.10, 0.20]  # relative to dataset default


def load_dataset(dataset, data_dir="datasets"):
    offline_csv = os.path.join(data_dir, dataset, f"{dataset}_offline.csv")
    online_csv = os.path.join(data_dir, dataset, f"{dataset}_online.csv")
    gt_csv = os.path.join(data_dir, dataset, f"{dataset}_2k.log_structured.csv")

    df = pd.read_csv(offline_csv)
    if os.path.exists(online_csv):
        df = pd.concat([df, pd.read_csv(online_csv)], ignore_index=True)
    gt_df = pd.read_csv(gt_csv)
    return df, gt_df


def get_embeddings(dataset, unique_logs, bert_model="all-MiniLM-L6-v2",
                    cache_dir="results/sensitivity_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset}_embeddings.npy")
    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if emb.shape[0] == len(unique_logs):
            print(f"  [{dataset}] Using cached embeddings ({emb.shape})")
            return emb
    from sentence_transformers import SentenceTransformer
    print(f"  [{dataset}] Computing embeddings ({len(unique_logs)} unique logs)...")
    model = SentenceTransformer(bert_model)
    emb = model.encode(unique_logs, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    np.save(cache_path, emb)
    return emb


def build_ground_truth_and_mapping(df, gt_df, unique_logs):
    """Maps every original log row to (a) its unique_log index and
    (b) its true EventId/EventTemplate via line_id -> LineId join."""
    unique_index = {u: i for i, u in enumerate(unique_logs)}
    gt_event_id = gt_df.set_index("LineId")["EventId"].to_dict()
    gt_event_tpl = gt_df.set_index("LineId")["EventTemplate"].to_dict()
    gt_content = gt_df.set_index("LineId")["Content"].to_dict()

    rows = []
    for _, row in df.iterrows():
        norm = row.get("normalised")
        lid = row.get("line_id")
        if pd.isna(norm) or norm not in unique_index:
            continue
        if lid not in gt_event_id:
            continue
        rows.append({
            "unique_idx": unique_index[norm],
            "gt_event_id": str(gt_event_id[lid]),
            "gt_template": str(gt_event_tpl[lid]),
            "gt_content": str(gt_content[lid]),
        })
    return pd.DataFrame(rows)


def evaluate_variant(mapping_df, labels, templates_map):
    """labels: cluster label per unique_log index.
       templates_map: cluster_id -> template string."""
    pred_templates = [templates_map.get(labels[r.unique_idx], "<*>")
                       for r in mapping_df.itertuples()]
    pred_ids = [str(labels[r.unique_idx]) for r in mapping_df.itertuples()]
    gt_ids = mapping_df["gt_event_id"].tolist()
    gt_templates = mapping_df["gt_template"].tolist()
    contents = mapping_df["gt_content"].tolist()

    GA = compute_GA(gt_ids, pred_ids)
    PA = compute_PA(gt_templates, pred_templates, contents)
    FGA = compute_FGA(gt_ids, pred_ids)
    FTA = compute_FTA(gt_ids, pred_ids, gt_templates, pred_templates)
    return GA, PA, FGA, FTA


def run_sweep(dataset, sweep):
    if dataset not in DATASET_PARAMS:
        print(f"[{dataset}] not in DATASET_PARAMS, skipping")
        return []

    default_dist, default_tau, n_clusters = DATASET_PARAMS[dataset]
    df, gt_df = load_dataset(dataset)
    unique_logs = df["normalised"].dropna().unique().tolist()
    raw_contents = []
    for u in unique_logs:
        match = df[df["normalised"] == u]["content"].iloc[0]
        raw_contents.append(match)

    embeddings = get_embeddings(dataset, unique_logs)
    mapping_df = build_ground_truth_and_mapping(df, gt_df, unique_logs)
    print(f"  [{dataset}] {len(mapping_df)} logs mapped to ground truth "
          f"(of {len(df)} total)")

    results = []

    if sweep in ("tau_d", "both"):
        print(f"\n  === Sweeping tau_d (default={default_dist}) ===")
        for delta in TAU_D_DELTA:
            dist = round(min(0.99, max(0.05, default_dist + delta)), 3)
            labels = two_stage_cluster(embeddings, unique_logs, dist,
                                        dataset=dataset, n_clusters=n_clusters)
            templates_map = extract_templates(unique_logs, raw_contents, labels, default_tau)
            GA, PA, FGA, FTA = evaluate_variant(mapping_df, labels, templates_map)
            tag = "default" if delta == 0.0 else f"{delta:+.2f}"
            print(f"    tau_d={dist:.3f} ({tag:8s})  GA={GA:.4f}  PA={PA:.4f}  "
                  f"FGA={FGA:.4f}  FTA={FTA:.4f}  n_clusters={len(set(labels))}")
            results.append({"dataset": dataset, "param": "tau_d", "value": dist,
                             "GA": round(GA, 4), "PA": round(PA, 4),
                             "FGA": round(FGA, 4), "FTA": round(FTA, 4)})

    if sweep in ("tau_r", "both"):
        print(f"\n  === Sweeping tau_r (default={default_tau}), "
              f"fixed clustering at tau_d={default_dist} ===")
        labels = two_stage_cluster(embeddings, unique_logs, default_dist,
                                    dataset=dataset, n_clusters=n_clusters)
        for tau_r in TAU_R_VALUES:
            templates_map = extract_templates(unique_logs, raw_contents, labels, tau_r)
            GA, PA, FGA, FTA = evaluate_variant(mapping_df, labels, templates_map)
            tag = "default" if abs(tau_r - default_tau) < 1e-6 else ""
            print(f"    tau_r={tau_r:.2f} {tag:8s}  GA={GA:.4f}  PA={PA:.4f}  "
                  f"FGA={FGA:.4f}  FTA={FTA:.4f}")
            results.append({"dataset": dataset, "param": "tau_r", "value": tau_r,
                             "GA": round(GA, 4), "PA": round(PA, 4),
                             "FGA": round(FGA, 4), "FTA": round(FTA, 4)})

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--sweep", type=str, default="both",
                     choices=["tau_r", "tau_d", "both"])
    args = ap.parse_args()

    results = run_sweep(args.dataset, args.sweep)
    if results:
        df_out = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        out_path = f"results/tau_sensitivity_{args.dataset}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
