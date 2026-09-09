import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.evaluate import evaluate_dataset

DATASETS = ["HDFS", "Hadoop", "Spark", "Zookeeper", "BGL", "HPC",
            "Thunderbird", "Windows", "Linux", "Android", "HealthApp",
            "Apache", "Proxifier", "OpenSSH", "OpenStack", "Mac"]

ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def sweep_dataset(dataset):
    rows = []
    for a in ALPHA_VALUES:
        print(f"  [{dataset}] testing alpha={a} ...")
        m = evaluate_dataset(dataset, alpha=a)
        if m is None:
            print(f"  [{dataset}] evaluate_dataset returned None, skipping")
            continue
        rows.append({
            "dataset": dataset,
            "alpha": a,
            "GA": m["GA"],
            "PA": m["PA"],
            "FGA": m["FGA"],
            "FTA": m["FTA"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not datasets:
        ap.print_help()
        return

    all_rows = []
    best_per_dataset = {}

    for ds in datasets:
        rows = sweep_dataset(ds)
        all_rows.extend(rows)
        if rows:
            df_ds = pd.DataFrame(rows)
            best_row = df_ds.loc[df_ds["GA"].idxmax()]
            best_per_dataset[ds] = best_row["alpha"]
            print(f"  [{ds}] BEST alpha by GA = {best_row['alpha']} (GA={best_row['GA']:.4f}, FGA={best_row['FGA']:.4f})")
            print(f"  [{ds}] full sweep: " + "  ".join(f"a={r['alpha']}:GA={r['GA']:.3f}" for r in rows))

    if all_rows:
        df = pd.DataFrame(all_rows)
        os.makedirs("results", exist_ok=True)
        out_path = "results/alpha_sweep_full.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved full sweep results to {out_path}")

        print("\n" + "=" * 60)
        print("BEST ALPHA PER DATASET (by GA):")
        for ds, a in best_per_dataset.items():
            print(f"  {ds:12s}: alpha={a}")


if __name__ == "__main__":
    main()
