"""
evaluation/evaluate.py
================================
Step 9: Evaluate offline parsing for all 16 datasets.
Computes: F1, Accuracy, GA, PA, FGA, FTA

Usage
-----
    python evaluation/evaluate.py --dataset HDFS
    python evaluation/evaluate.py --all
"""

import os
import sys
import json
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DATASETS = ["HDFS","Hadoop","Spark","Zookeeper","BGL","HPC","Thunderbird",
            "Windows","Linux","Android","HealthApp","Apache","Proxifier",
            "OpenSSH","OpenStack","Mac"]


def evaluate_dataset(
    dataset:     str,
    groundtruth: str,
    parsedresult: str,
) -> dict:
    """Run all 6 metrics for one dataset."""
    # Import the evaluator from logparser/utils
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from logparser.utils import evaluator as ev
        metrics = ev.evaluate_all(groundtruth, parsedresult, verbose=True)
    except ImportError:
        # Fallback: basic F1 + accuracy only
        df_gt = pd.read_csv(groundtruth)
        df_pd = pd.read_csv(parsedresult)
        correct = (df_gt["EventTemplate"].astype(str) ==
                   df_pd["EventTemplate"].astype(str)).sum()
        metrics = {
            "F1_measure": correct / len(df_gt),
            "Accuracy":   correct / len(df_gt),
            "GA": 0.0, "PA": 0.0, "FGA": 0.0, "FTA": 0.0,
        }

    metrics["dataset"] = dataset
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate offline parsing")
    parser.add_argument("--dataset",    type=str,  default=None)
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--data_dir",   type=str,  default="../../data/loghub_2k")
    parser.add_argument("--result_dir", type=str,  default="../../logparser/ParseXFormer/ParseXFormer_result")
    parser.add_argument("--output_dir", type=str,  default="results")
    args = parser.parse_args()

    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not datasets:
        parser.print_help(); sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = []

    for ds in datasets:
        gt   = os.path.join(args.data_dir,   ds, f"{ds}_2k.log_structured.csv")
        pred = os.path.join(args.result_dir, f"{ds}_2k.log_structured.csv")

        if not os.path.exists(gt) or not os.path.exists(pred):
            print(f"[{ds}] Files not found — skipping.")
            continue

        print(f"\n{'='*50}\n  {ds}\n{'='*50}")
        m = evaluate_dataset(ds, gt, pred)
        all_metrics.append(m)

    if all_metrics:
        df = pd.DataFrame(all_metrics).set_index("dataset")
        METRICS = ["F1_measure","Accuracy","GA","PA","FGA","FTA"]
        df_out = df[[c for c in METRICS if c in df.columns]]

        print("\n" + "="*70)
        print("OFFLINE EVALUATION RESULTS — ALL DATASETS")
        print("="*70)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(df_out.to_string())
        print("\nMean:")
        print(df_out.mean().to_string())

        df_out.to_csv(os.path.join(args.output_dir, "offline_metrics.csv"),
                      float_format="%.6f")
        print(f"\nSaved: {args.output_dir}/offline_metrics.csv")


if __name__ == "__main__":
    main()
