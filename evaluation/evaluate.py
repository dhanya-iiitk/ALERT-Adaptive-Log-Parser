"""
evaluation/evaluate.py
========================
Evaluate offline parsing for all 16 datasets.
Uses single SBERT model: all-MiniLM-L6-v2
Per-dataset alpha tuning for best results.

Usage
-----
    python3 evaluation/evaluate.py --dataset HDFS
    python3 evaluation/evaluate.py --all
"""

import os, sys, re, json, argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocess import DATASET_CONFIG

DATASETS = ["HDFS","Hadoop","Spark","Zookeeper","BGL","HPC","Thunderbird",
            "Windows","Linux","Android","HealthApp","Apache","Proxifier",
            "OpenSSH","OpenStack","Mac"]

DATASET_ALPHA = {
    "HDFS":0.7,"Hadoop":0.7,"Spark":0.7,"Zookeeper":0.7,
    "BGL":0.7,"HPC":0.7,"Thunderbird":0.7,"Windows":0.7,
    "Linux":0.7,"Android":0.7,"HealthApp":0.2,"Apache":0.3,
    "Proxifier":0.7,"OpenSSH":0.7,"OpenStack":0.7,"Mac":0.7,
}

def load_store(store_path):
    with open(store_path) as f: data = json.load(f)
    for e in data:
        e["_centroid"] = np.array(e["centroid"], dtype=np.float32)
        toks = e["template"].split()
        e["_fixed"]    = [t for t in toks if t != "<*>"]
        e["_first_kw"] = toks[0].lower() if toks else ""
    return data

def embed_logs(logs, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {model_name} ...")
    model = SentenceTransformer(model_name)
    return model.encode(logs, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)

def structural_score(log_toks, entry):
    fixed = entry["_fixed"]
    if not fixed: return 0.3
    log_set = set(t.lower() for t in log_toks)
    matches = sum(1 for t in fixed if t.lower() in log_set)
    token_score = matches / len(fixed)
    first_kw = 1.0 if (log_toks and log_toks[0].lower()==entry["_first_kw"]) else 0.0
    tmpl_len = len(entry["template"].split())
    log_len  = len(log_toks)
    len_sim  = 1.0 - abs(log_len-tmpl_len)/max(log_len,tmpl_len,1)
    return 0.5*token_score + 0.3*first_kw + 0.2*len_sim

def match_logs(embeddings, store, normalised_logs, alpha=0.5):
    centroids  = np.stack([e["_centroid"] for e in store])
    cos_scores = embeddings @ centroids.T
    pred_ids, pred_tmpls = [], []
    for i, norm_log in enumerate(normalised_logs):
        log_toks = norm_log.split()
        struct   = np.array([structural_score(log_toks, e) for e in store])
        hybrid   = alpha * cos_scores[i] + (1-alpha) * struct
        best_j   = int(np.argmax(hybrid))
        pred_ids.append(store[best_j]["template_id"])
        pred_tmpls.append(store[best_j]["template"])
    return pred_ids, pred_tmpls

def normalise_template(template):
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
    for i,(g,p) in enumerate(zip(gt_ids,pred_ids)):
        gt_g[g].add(i); pd_g[p].add(i)
    correct = sum(1 for i in range(len(gt_ids))
                  if gt_g[gt_ids[i]]==pd_g[pred_ids[i]])
    return correct/len(gt_ids)

def compute_PA(gt_templates, pred_templates, contents):
    def label(log_toks, tmpl_toks):
        labels=[]; t=l=0
        while l<len(log_toks):
            if t>=len(tmpl_toks): labels.append(False); l+=1; continue
            if tmpl_toks[t]=="<*>":
                nf=next((tmpl_toks[k] for k in range(t+1,len(tmpl_toks))
                         if tmpl_toks[k]!="<*>"),None)
                if nf is None:
                    while l<len(log_toks): labels.append(True); l+=1
                else:
                    while l<len(log_toks) and log_toks[l]!=nf:
                        labels.append(True); l+=1
                t+=1
            else: labels.append(False); l+=1; t+=1
        return labels
    correct=0
    for gt_t,pd_t,content in zip(gt_templates,pred_templates,contents):
        if label(content.split(),normalise_template(gt_t).split())==\
           label(content.split(),normalise_template(pd_t).split()):
            correct+=1
    return correct/max(len(gt_templates),1)

def compute_FGA(gt_ids, pred_ids):
    gt_g,pd_g=defaultdict(set),defaultdict(set)
    for i,(g,p) in enumerate(zip(gt_ids,pred_ids)):
        gt_g[g].add(i); pd_g[p].add(i)
    gt_sets={frozenset(v) for v in gt_g.values()}
    pd_list=[frozenset(v) for v in pd_g.values()]
    pd_set=set(pd_list)
    PGA=sum(1 for ps in pd_list if ps in gt_sets)/max(len(pd_list),1)
    RGA=sum(1 for gs in gt_sets if gs in pd_set)/max(len(gt_sets),1)
    return 2*PGA*RGA/(PGA+RGA) if (PGA+RGA)>0 else 0.0

def compute_FTA(gt_ids, pred_ids, gt_templates, pred_templates):
    gt_n=[normalise_template(t) for t in gt_templates]
    pd_n=[normalise_template(t) for t in pred_templates]
    pd_g=defaultdict(lambda:{"gt":[],"pd":None})
    for i in range(len(pred_ids)):
        pd_g[pred_ids[i]]["gt"].append(gt_n[i])
        pd_g[pred_ids[i]]["pd"]=pd_n[i]
    correct_set=set(); pd_correct=0
    for eid,info in pd_g.items():
        maj=Counter(info["gt"]).most_common(1)[0][0]
        if info["pd"]==maj: pd_correct+=1; correct_set.add(maj)
    unique_gt=set(gt_n)
    PTA=pd_correct/max(len(pd_g),1)
    RTA=len(correct_set)/max(len(unique_gt),1)
    return 2*PTA*RTA/(PTA+RTA) if (PTA+RTA)>0 else 0.0

def evaluate_dataset(dataset, data_dir="datasets", store_dir="repository",
                     model="all-MiniLM-L6-v2", alpha=None):
    if alpha is None:
        alpha = DATASET_ALPHA.get(dataset, 0.5)

    gt_path    = os.path.join(data_dir, dataset,
                              f"{dataset}_2k.log_structured.csv")
    store_path = os.path.join(store_dir,
                              f"{dataset}_template_store.json")

    if not os.path.exists(gt_path):
        print(f"  [{dataset}] Ground truth not found"); return None
    if not os.path.exists(store_path):
        print(f"  [{dataset}] Store not found"); return None

    print(f"\n{'='*50}\n  Evaluating: {dataset}  (alpha={alpha})\n{'='*50}")

    df_gt     = pd.read_csv(gt_path)
    non_empty = df_gt[~df_gt["EventId"].isnull()].index
    df_gt     = df_gt.loc[non_empty].reset_index(drop=True)

    gt_ids       = df_gt["EventId"].astype(str).tolist()
    gt_templates = df_gt["EventTemplate"].astype(str).tolist()
    contents     = df_gt["Content"].astype(str).tolist()

    cfg = DATASET_CONFIG[dataset]
    rex = [re.compile(r) for r in cfg["regex"]]
    normalised = []
    for c in contents:
        n = c
        for r in rex: n = r.sub("<*>", n)
        normalised.append(re.sub(r"\s+", " ", n).strip())

    store      = load_store(store_path)
    print(f"  Store: {len(store)} templates")
    embeddings = embed_logs(normalised, model)

    pred_ids, pred_templates = match_logs(
        embeddings, store, normalised, alpha)

    GA  = compute_GA(gt_ids, pred_ids)
    PA  = compute_PA(gt_templates, pred_templates, contents)
    FGA = compute_FGA(gt_ids, pred_ids)
    FTA = compute_FTA(gt_ids, pred_ids, gt_templates, pred_templates)

    metrics = {
        "dataset": dataset,
        "GA":  round(GA,  4), "PA":  round(PA,  4),
        "FGA": round(FGA, 4), "FTA": round(FTA, 4),
        "alpha": alpha, "store_templates": len(store),
    }
    print(f"  GA={GA:.4f}  PA={PA:.4f}  FGA={FGA:.4f}  FTA={FTA:.4f}")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   type=str,   default=None)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--data_dir",  type=str,   default="datasets")
    parser.add_argument("--store_dir", type=str,   default="repository")
    parser.add_argument("--model",     type=str,   default="all-MiniLM-L6-v2")
    parser.add_argument("--alpha",     type=float, default=None,
                        help="Override per-dataset alpha (default: use DATASET_ALPHA)")
    args = parser.parse_args()

    datasets = DATASETS if args.all else (
        [args.dataset] if args.dataset else [])
    if not datasets:
        parser.print_help(); sys.exit(1)

    os.makedirs("results", exist_ok=True)
    all_metrics = []

    for ds in datasets:
        m = evaluate_dataset(
            ds, args.data_dir, args.store_dir,
            args.model, args.alpha)
        if m: all_metrics.append(m)

    if all_metrics:
        df = pd.DataFrame(all_metrics).set_index("dataset")
        METRICS = ["GA","PA","FGA","FTA"]
        print("\n"+"="*60)
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
