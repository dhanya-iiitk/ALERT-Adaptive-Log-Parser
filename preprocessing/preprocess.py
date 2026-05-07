"""
preprocessing/preprocess.py
============================
Step 1-5: Preprocess all 16 datasets independently.
- Extract Content field
- Apply variable normalisation
- Remove duplicates
- Create 70/30 offline/online split

Usage
-----
    python preprocessing/preprocess.py --dataset HDFS
    python preprocessing/preprocess.py --all
"""

import re
import os
import sys
import json
import argparse
import pandas as pd
from typing import List, Tuple, Dict

# ── Dataset configurations (log_format + regex per dataset) ───────────────

DATASET_CONFIG = {
    "HDFS": {
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
    },
    "Hadoop": {
        "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        "regex": [r"(\d+\.){3}\d+"],
    },
    "Spark": {
        "log_format": "<Date> <Time> <Level> <Component>: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+"],
    },
    "Zookeeper": {
        "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
        "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
    },
    "BGL": {
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "regex": [r"core\.\d+"],
    },
    "HPC": {
        "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        "regex": [r"=\d+"],
    },
    "Thunderbird": {
        "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
        "regex": [r"(\d+\.){3}\d+"],
    },
    "Windows": {
        "log_format": "<Date> <Time>, <Level>                  <Component>    <Content>",
        "regex": [r"0x.*?\s"],
    },
    "Linux": {
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}"],
    },
    "Android": {
        "log_format": "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
        "regex": [r"(/[\w-]+)+", r"([\w-]+\.){2,}[\w-]+",
                  r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b"],
    },
    "HealthApp": {
        "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
        "regex": [],
    },
    "Apache": {
        "log_format": "\[<Time>\] \[<Level>\] <Content>",
        "regex": [r"(\d+\.){3}\d+"],
    },
    "Proxifier": {
        "log_format": "\[<Time>\] <Program> - <Content>",
        "regex": [r"<\d+\s?sec", r"([\w-]+\.)+[\w-]+(:\d+)?",
                  r"\d{2}:\d{2}(:\d{2})*", r"[KGTM]B"],
    },
    "OpenSSH": {
        "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"],
    },
    "OpenStack": {
        "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        "regex": [r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"],
    },
    "Mac": {
        "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        "regex": [r"([\w-]+\.){2,}[\w-]+"],
    },
}

SPLIT_RATIO = 0.70   # 70% offline, 30% online


# ── Core functions ─────────────────────────────────────────────────────────

def compile_format(log_format: str):
    headers, regex_str = [], ""
    for k, part in enumerate(re.split(r"(<[^<>]+>)", log_format)):
        if k % 2 == 0:
            regex_str += part.replace(" ", r"\s+")
        else:
            h = part.strip("<").strip(">")
            regex_str += f"(?P<{h}>.*?)"
            headers.append(h)
    return headers, re.compile("^" + regex_str + "$")


def extract_content(line: str, fmt_re, headers: List[str]) -> str:
    if fmt_re is None:
        return line.strip()
    m = fmt_re.search(line.strip())
    if m and "Content" in m.groupdict():
        return m.group("Content").strip()
    return line.strip()


def normalise(content: str, rex_patterns: List) -> str:
    log = content
    for rex in rex_patterns:
        log = re.sub(rex, "<*>", log)
    return re.sub(r"\s+", " ", log).strip()


def preprocess_dataset(
    dataset:    str,
    data_dir:   str = "datasets",
    out_dir:    str = "datasets",
    split:      float = SPLIT_RATIO,
) -> Dict:
    """
    Steps 1–5 for one dataset.
    Reads raw log file, outputs:
      - {dataset}/{dataset}_offline.csv  (70%)
      - {dataset}/{dataset}_online.csv   (30%)
      - {dataset}/{dataset}_meta.json
    """
    cfg      = DATASET_CONFIG[dataset]
    log_file = os.path.join(data_dir, dataset, f"{dataset}_2k.log")
    if not os.path.exists(log_file):
        print(f"[{dataset}] Log file not found: {log_file} — skipping.")
        return {}

    headers, fmt_re = compile_format(cfg["log_format"])
    rex_patterns    = [re.compile(r) for r in cfg["regex"]]

    print(f"[{dataset}] Reading {log_file} ...")
    with open(log_file, encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    # Step 1.2 + 1.4: extract content and normalise
    records = []
    for i, line in enumerate(raw_lines):
        content = extract_content(line, fmt_re, headers)
        if not content:
            continue
        norm = normalise(content, rex_patterns)
        records.append({
            "line_id":    i + 1,
            "raw":        line.strip(),
            "content":    content,
            "normalised": norm,
        })

    df = pd.DataFrame(records)

    # Step 1.3: mark duplicates (keep all rows, tag duplicates)
    df["is_duplicate"] = df["normalised"].duplicated(keep="first")

    # Step 1.5: create 70/30 split
    n       = len(df)
    n_off   = int(n * split)
    df_off  = df.iloc[:n_off].copy()
    df_on   = df.iloc[n_off:].copy()

    # Save
    ds_out = os.path.join(out_dir, dataset)
    os.makedirs(ds_out, exist_ok=True)

    off_path = os.path.join(ds_out, f"{dataset}_offline.csv")
    on_path  = os.path.join(ds_out, f"{dataset}_online.csv")
    df_off.to_csv(off_path,  index=False)
    df_on.to_csv(on_path,    index=False)

    meta = {
        "dataset":          dataset,
        "total_logs":       n,
        "offline_logs":     len(df_off),
        "online_logs":      len(df_on),
        "unique_offline":   int(df_off["normalised"].nunique()),
        "split_ratio":      split,
        "offline_file":     off_path,
        "online_file":      on_path,
        "log_format":       cfg["log_format"],
        "regex_count":      len(cfg["regex"]),
    }
    with open(os.path.join(ds_out, f"{dataset}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{dataset}] Total={n}  Offline={len(df_off)}  Online={len(df_on)}  "
          f"UniqueOffline={meta['unique_offline']}")
    return meta


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess log datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset name (e.g. HDFS)")
    parser.add_argument("--all",     action="store_true",
                        help="Process all 16 datasets")
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Root directory of raw log files")
    parser.add_argument("--split",   type=float, default=SPLIT_RATIO,
                        help="Offline train split ratio (default 0.70)")
    args = parser.parse_args()

    datasets = list(DATASET_CONFIG.keys()) if args.all else [args.dataset]

    if not datasets or datasets == [None]:
        parser.print_help(); sys.exit(1)

    all_meta = {}
    for ds in datasets:
        if ds not in DATASET_CONFIG:
            print(f"Unknown dataset: {ds}"); continue
        meta = preprocess_dataset(ds, args.data_dir, args.data_dir, args.split)
        if meta:
            all_meta[ds] = meta

    # Summary
    if all_meta:
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"  {'Dataset':<14} {'Total':>7} {'Offline':>9} {'Online':>7} {'Unique':>7}")
        print("  " + "-" * 48)
        for ds, m in all_meta.items():
            print(f"  {ds:<14} {m['total_logs']:>7} {m['offline_logs']:>9} "
                  f"{m['online_logs']:>7} {m['unique_offline']:>7}")
        with open("results/preprocessing_summary.json", "w") as f:
            json.dump(all_meta, f, indent=2)
        print("\nSaved: results/preprocessing_summary.json")


if __name__ == "__main__":
    main()
