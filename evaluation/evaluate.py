"""
evaluation/evaluate.py
========================
Step 9: Evaluate offline parsing for all 16 datasets.

How it works:
1. Load ground truth from datasets/{DS}/{DS}_2k.log_structured.csv
2. Load our template store from repository/{DS}_template_store.json
3. Embed all logs using SBERT
4. Match each log to nearest template via cosine similarity
5. Compute GA, PA, FGA, FTA metrics

Usage
-----
    python3 evaluation/evaluate.py --dataset HDFS
    python3 evaluation/evaluate.py --all
"""

import os
import sys
import json
import re
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocess import DATASET_CONFIG

DATASETS = ["HDFS","Hadoop","Spark","Zookeeper","BGL","HPC","Thunderbird",
            "Windows","Linux","Android","HealthApp","Apache","Proxifier",
            "OpenSSH","OpenStack","Mac"]


def load_store(store_path: str) -> List[Dict]:
    with open(store_path) as f:
        data = json.load(f)
    for e in data:
        e["_centroid"] = np.array(e["centroid"], dtype=np.float32)
    return data


def embed_logs(logs: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        logs, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def match_logs_to_templates(
    embeddings: np.ndarray,
    store:      List[Dict],
) -> Tuple[List[str], List[str]]:
    centroids  = np.stack([e["_centroid"] for e in store])
    scores     = embeddings @ centroids.T
    best_idx   = np.argmax(scores, axis=1)
    pred_ids   = [store[i]["template_id"] for i in best_idx]
    pred_tmpls = [store[i]["template"]    for i in best_idx]
    return pred_ids, pred_tmpls


def normalise_template(template: str) -> str:
    tokens = template.strip().split()
    result, prev = [], False
    for tok in tokens:
        if "<*>" in tok:
            if not prev: result.append("<*>")
            prev = True
        else:
            result.append(tok); prev = False
    return " ".join(result)


def compute_GA(gt_ids, pred_ids):
    gt_g, pd_g = defaultdict(set), defaultdict(set)
    for i,(g,p) in enumerate(zip(gt_ids, pred_ids)):
        gt_g[g].add(i); pd_g[p].add(i)
    correct = sum(1 for i in range(len(gt_ids))
                  if gt_g[gt_ids[i]] == pd_g[pred_ids[i]])
    return correct / len(gt_ids)


def compute_PA(gt_templates, pred_templates, contents):
    def label(log_toks, tmpl_toks):
        labels=[]; t=l=0
        while l < len(log_toks):
            if t >= len(tmpl_toks):
                labels.append(False); l+=1; continue
            if tmpl_toks[t]=="<*>":
                nf = next((tmpl_toks[k] for k in range(t+1,len(tmpl_toks))
                           if tmpl_toks[k]!="<*>"), None)
                if nf is None:
                    while l<len(log_toks): labels.append(True); l+=1
                else:
                    while l<len(log_toks) and log_toks[l]!=nf:
                        labels.append(True); l+=1
                t+=1
            else:
                labels.append(False); l+=1; t+=1
        return labels

    correct = 0
    for gt_t, pd_t, content in zip(gt_templates, pred_templates, contents):
        gt_n = normalise_template(gt_t).split()
        pd_n = normalise_template(pd_t).split()
        log_toks = content.strip().split()
        if label(log_toks, gt_n) == label(log_toks, pd_n):
            correct += 1
    return correct / max(len(gt_templates), 1)


def compute_FGA(gt_ids, pred_ids):
    gt_g, pd_g = defaultdict(set), defaultdict(set)
    for i,(g,p) in enumerate(zip(gt_ids, pred_ids)):
        gt_g[g].add(i); pd_g[p].add(i)
    gt_sets = {frozenset(v) for v in gt_g.values()}
    pd_list = [frozenset(v) for v in pd_g.values()]
    pd_set  = set(pd_list)
    PGA = sum(1 for ps in pd_list if ps in gt_sets) / max(len(pd_list),1)
    RGA = sum(1 for gs in gt_sets if gs in pd_set)  / max(len(gt_sets), 1)
    FGA = 2*PGA*RGA/(PGA+RGA) if (PGA+RGA)>0 else 0.0
    return PGA, RGA, FGA


def compute_FTA(gt_ids, pred_ids, gt_templates, pred_templates):
    gt_n  = [normalise_template(t) for t in gt_templates]
    pd_n  = [normalise_template(t) for t in pred_templates]
    pd_g  = defaultdict(lambda: {"gt":[], "pd":None})
    for i in range(len(pred_ids)):
        pd_g[pred_ids[i]]["gt"].append(gt_n[i])
        pd_g[pred_ids[i]]["pd"] = pd_n[i]
    correct_set = set(); pd_correct = 0
    for eid, info in pd_g.items():
        maj = Counter(info["gt"]).most_common(1)[0][0]
        if info["pd"] == maj:
            pd_correct += 1; correct_set.add(maj)
    unique_gt = set(gt_n)
    PTA = pd_correct / max(len(pd_g), 1)
    RTA = len(correct_set) / max(len(unique_gt), 1)
    FTA = 2*PTA*RTA/(PTA+RTA) if (PTA+RTA)>0 else 0.0
    return PTA, RTA, FTA


def evaluate_dataset(dataset, data_dir="datasets", store_dir="repository",
                     model="all-MiniLM-L6-v2"):
    gt_path    = os.path.join(data_dir, dataset, f"{dataset}_2k.log_structured.csv")
    store_path = os.path.join(store_dir, f"{dataset}_template_store.json")

    if not os.path.exists(gt_path):
        print(f"  [{dataset}] Ground truth not found: {gt_path}"); return None
    if not os.path.exists(store_path):
        print(f"  [{dataset}] Store not found — run build_store.py first"); return None

    print(f"\n{'='*50}\n  Evaluating: {dataset}\n{'='*50}")
    df_gt    = pd.read_csv(gt_path)
    non_empty = df_gt[~df_gt["EventId"].isnull()].index
    df_gt    = df_gt.loc[non_empty].reset_index(drop=True)

    gt_ids       = df_gt["EventId"].astype(str).tolist()
    gt_templates = df_gt["EventTemplate"].astype(str).tolist()
    contents     = df_gt["Content"].astype(str).tolist()

    # Preprocess using dataset regex
    cfg = DATASET_CONFIG[dataset]
    rex = [re.compile(r) for r in cfg["regex"]]
    normalised = []
    for c in contents:
        n = c
        for r in rex: n = r.sub("<*>", n)
        normalised.append(re.sub(r"\s+", " ", n).strip())

    store      = load_store(store_path)
    print(f"  Store: {len(store)} templates | Logs: {len(normalised)}")
    embeddings = embed_logs(normalised, model)
    pred_ids, pred_templates = match_logs_to_templates(embeddings, store)

    GA          = compute_GA(gt_ids, pred_ids)
    PA          = compute_PA(gt_templates, pred_templates, contents)
    PGA,RGA,FGA = compute_FGA(gt_ids, pred_ids)
    PTA,RTA,FTA = compute_FTA(gt_ids, pred_ids, gt_templates, pred_templates)

    metrics = {
        "dataset": dataset,
        "GA":  round(GA,  4), "PA":  round(PA,  4),
        "FGA": round(FGA, 4), "FTA": round(FTA, 4),
        "PGA": round(PGA, 4), "RGA": round(RGA, 4),
        "PTA": round(PTA, 4), "RTA": round(RTA, 4),
        "store_templates": len(store),
    }
    print(f"  GA={GA:.4f}  PA={PA:.4f}  FGA={FGA:.4f}  FTA={FTA:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   type=str, default=None)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--data_dir",  type=str, default="datasets")
    parser.add_argument("--store_dir", type=str, default="repository")
    parser.add_argument("--model",     type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not datasets:
        parser.print_help(); sys.exit(1)

    os.makedirs("results", exist_ok=True)
    all_metrics = []

    for ds in datasets:
        m = evaluate_dataset(ds, args.data_dir, args.store_dir, args.model)
        if m: all_metrics.append(m)

    if all_metrics:
        df = pd.DataFrame(all_metrics).set_index("dataset")
        METRICS = ["GA","PA","FGA","FTA"]
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(df[METRICS].to_string())
        print("\nMean:")
        for m in METRICS:
            print(f"  {m}: {df[m].mean():.4f}")
        df.to_csv("results/evaluation_metrics.csv", float_format="%.6f")
        with open("results/evaluation_metrics.json","w") as f:
            json.dump(all_metrics, f, indent=2)
        print("\nSaved: results/evaluation_metrics.csv")


if __name__ == "__main__":
    main()
