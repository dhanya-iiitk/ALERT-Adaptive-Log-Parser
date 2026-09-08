"""
compute_online_metrics.py  (CORRECTED join logic)

Computes GA, PA, FGA, FTA for ALERT's ONLINE/STREAMING results, using the
exact same functions already defined and trusted in evaluation/evaluate.py.

IMPORTANT: `log_id` in <DATASET>_stream_results.csv is just the 1-indexed
POSITION of each log in the stream - it is NOT the same as the original
`LineId` in the ground-truth file. The true LineId must be looked up via
the corresponding row in datasets/<DATASET>/<DATASET>_online.csv, which
has its own `line_id` column carrying the *original* LineId.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.evaluate import compute_GA, compute_PA, compute_FGA, compute_FTA  # noqa: E402

DATASETS = ["HDFS", "Hadoop", "Spark", "Zookeeper", "BGL", "HPC",
            "Thunderbird", "Windows", "Linux", "Android", "HealthApp",
            "Apache", "Proxifier", "OpenSSH", "OpenStack", "Mac"]


def compute_online_metrics(dataset, results_dir="results",
                            data_dir="datasets", temporal=False):
    res_dir = os.path.join(results_dir, "temporal_streaming") if temporal else results_dir
    res_path = os.path.join(res_dir, f"{dataset}_stream_results.csv")
    online_path = os.path.join(data_dir, dataset, f"{dataset}_online.csv")
    gt_path = os.path.join(data_dir, dataset, f"{dataset}_2k.log_structured.csv")

    for p in (res_path, online_path, gt_path):
        if not os.path.exists(p):
            print(f"  [{dataset}] missing {p}, skipping"); return None

    df_res = pd.read_csv(res_path).reset_index(drop=True)
    df_online = pd.read_csv(online_path).reset_index(drop=True)
    df_gt = pd.read_csv(gt_path)

    if len(df_res) != len(df_online):
        print(f"  [{dataset}] WARNING: row count mismatch - "
              f"stream_results has {len(df_res)}, online.csv has {len(df_online)}. "
              f"Position-based join may be unsafe; check manually.")

    n = min(len(df_res), len(df_online))
    real_line_ids = df_online["line_id"].iloc[:n].tolist()

    df_res = df_res.iloc[:n].copy()
    df_res["real_line_id"] = real_line_ids

    gt_event_id = df_gt.set_index("LineId")["EventId"].to_dict()
    gt_event_tpl = df_gt.set_index("LineId")["EventTemplate"].to_dict()
    gt_content = df_gt.set_index("LineId")["Content"].to_dict()

    df_res["gt_event_id"] = df_res["real_line_id"].map(gt_event_id)
    df_res["gt_template"] = df_res["real_line_id"].map(gt_event_tpl)
    df_res["gt_content"] = df_res["real_line_id"].map(gt_content)

    before = len(df_res)
    df_res = df_res.dropna(subset=["gt_event_id", "gt_template", "gt_content"])
    dropped = before - len(df_res)
    if dropped:
        print(f"  [{dataset}] WARNING: {dropped} streamed logs had no "
              f"matching real_line_id in ground truth")

    gt_ids = df_res["gt_event_id"].astype(str).tolist()
    gt_templates = df_res["gt_template"].astype(str).tolist()
    contents = df_res["gt_content"].astype(str).tolist()
    pred_ids = df_res["top1_template"].astype(str).tolist()
    pred_templates = df_res["top1_template"].astype(str).tolist()

    GA = compute_GA(gt_ids, pred_ids)
    PA = compute_PA(gt_templates, pred_templates, contents)
    FGA = compute_FGA(gt_ids, pred_ids)
    FTA = compute_FTA(gt_ids, pred_ids, gt_templates, pred_templates)

    old_matched = (df_res["match_type"] != "new").sum()
    old_accuracy = old_matched / max(len(df_res), 1)

    return {
        "dataset": dataset,
        "total_logs": len(df_res),
        "GA": round(GA, 4),
        "PA": round(PA, 4),
        "FGA": round(FGA, 4),
        "FTA": round(FTA, 4),
        "old_1_minus_new_over_N": round(old_accuracy, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--temporal", action="store_true")
    args = ap.parse_args()

    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not datasets:
        ap.print_help()
        return

    rows = []
    for ds in datasets:
        r = compute_online_metrics(ds, temporal=args.temporal)
        if r:
            rows.append(r)
            print(f"  [{ds}] GA={r['GA']:.4f}  PA={r['PA']:.4f}  "
                  f"FGA={r['FGA']:.4f}  FTA={r['FTA']:.4f}   "
                  f"(old metric: {r['old_1_minus_new_over_N']:.4f})")

    if rows:
        df = pd.DataFrame(rows).set_index("dataset")
        print("\n" + "=" * 90)
        print(df.to_string())
        suffix = "_temporal" if args.temporal else ""
        out_path = f"results/online_metrics_corrected{suffix}.csv"
        df.to_csv(out_path)
        print(f"\nSaved corrected online metrics to {out_path}")


if __name__ == "__main__":
    main()
