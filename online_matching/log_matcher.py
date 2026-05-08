"""
online_matching/log_matcher.py
================================
Modules 7 + 8 — Real-Time Semantic Retrieval + Incremental Evolution

PHASE 3 Steps 11-15: Online streaming evaluation.

For a chosen dataset:
  - Load its {dataset}_template_store.json (built offline)
  - Stream the 30% unseen online logs one-by-one
  - Retrieve Top-3 nearest templates via cosine similarity (Step 7.4-7.5)
  - Make match decision: exact / partial / fuzzy / new (Step 7.6)
  - Update centroid via EMA if matched (Step 8.1)
  - Add new template if unmatched (Step 8.2)
  - Compact repository if over capacity (Step 8.3)
  - Report latency, throughput, retrieval accuracy, new templates discovered

Usage
-----
    python online_matching/log_matcher.py --dataset HDFS
    python online_matching/log_matcher.py --dataset BGL --top_k 3
"""

import os
import sys
import re
import json
import time
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import DATASET_CONFIG
from repository.template_store import TemplateStore, TemplateEntry, _make_id
from clustering.adaptive_cluster import EntropyThresholdController


# ─────────────────────────────────────────────────────────────────────────────
# Match result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """
    Step 7.5 output: one row per incoming log.

    Attributes
    ----------
    log_id          : sequential log index in the stream
    normalised      : preprocessed log content
    match_type      : 'exact' | 'partial' | 'fuzzy' | 'new'
    top1_template   : best matching template
    top1_score      : cosine similarity of best match (0-1)
    top2_template   : 2nd best (empty string if none)
    top2_score      : cosine similarity of 2nd best
    top3_template   : 3rd best
    top3_score      : cosine similarity of 3rd best
    latency_ms      : per-log wall-clock time
    store_size      : repository size after this log
    """
    log_id:          int
    normalised:      str
    match_type:      str
    top1_template:   str
    top1_score:      float
    top2_template:   str   = ""
    top2_score:      float = 0.0
    top3_template:   str   = ""
    top3_score:      float = 0.0
    latency_ms:      float = 0.0
    store_size:      int   = 0

    def __str__(self) -> str:
        lines = [
            f"  Log #{self.log_id}: {self.normalised[:60]}{'...' if len(self.normalised)>60 else ''}",
            f"  [Rank 1] {self.match_type.upper():7s}  score={self.top1_score:.4f}  "
            f"| {self.top1_template}",
        ]
        if self.top2_template:
            lines.append(
                f"  [Rank 2] {'':7s}  score={self.top2_score:.4f}  "
                f"| {self.top2_template}")
        if self.top3_template:
            lines.append(
                f"  [Rank 3] {'':7s}  score={self.top3_score:.4f}  "
                f"| {self.top3_template}")
        lines.append(f"  Latency: {self.latency_ms:.2f} ms  |  "
                     f"Store size: {self.store_size}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LogMatcher — Modules 7 + 8
# ─────────────────────────────────────────────────────────────────────────────

class LogMatcher:
    """
    Real-time log matcher for one dataset.

    Implements Module 7 (retrieval) and Module 8 (evolution).

    Parameters
    ----------
    dataset          : 'HDFS', 'BGL', etc.
    store_dir        : directory containing {dataset}_template_store.json
    bert_model       : BERT model name
    top_k            : number of top templates to retrieve (default 3)
    exact_thr        : cosine similarity >= this → 'exact'
    partial_thr      : cosine similarity >= this → 'partial'
    new_thr          : below this → create new template
    update_interval  : run entropy controller every N logs
    auto_save        : save store after every new template addition
    """

    def __init__(
        self,
        dataset:         str,
        store_dir:       str   = "repository",
        bert_model:      str   = "all-MiniLM-L6-v2",
        top_k:           int   = 3,
        exact_thr:       float = 0.99,
        partial_thr:     float = 0.80,
        new_thr:         float = 0.60,
        update_interval: int   = 100,
        auto_save:       bool  = True,
    ):
        self.dataset         = dataset
        self.top_k           = top_k
        self.exact_thr       = exact_thr
        self.partial_thr     = partial_thr
        self.new_thr         = new_thr
        self.update_interval = update_interval
        self.auto_save       = auto_save

        # Module 6 — load dataset-specific store
        self.store = TemplateStore(dataset=dataset, store_dir=store_dir)
        if len(self.store) == 0:
            raise FileNotFoundError(
                f"Store is empty: {self.store.store_path}\n"
                f"Run: python repository/build_store.py --dataset {dataset}")

        # Dataset config for preprocessing
        cfg = DATASET_CONFIG[dataset]
        self._rex     = [re.compile(r) for r in cfg["regex"]]
        self._fmt_re  = self._compile_format(cfg["log_format"])

        # Load BERT
        from sentence_transformers import SentenceTransformer
        self._sbert = SentenceTransformer(bert_model)
        print(f"[{dataset}] SBERT loaded | "
              f"Store: {len(self.store)} templates")

        # Module 3 — entropy-adaptive threshold
        self._ctrl      = EntropyThresholdController()
        self._log_count = 0

    # ── Public API ───────────────────────────────────────────────────

    def match(self, raw_log: str) -> List[MatchResult]:
        """
        Process one raw log line (Modules 7 + 8).

        Steps
        -----
        7.1  Log arrival
        7.2  Preprocessing
        7.3  BERT embedding
        7.4  Cosine similarity vs all centroids
        7.5  Top-K retrieval
        7.6  Match decision
        8.1/8.2  Update or add template

        Returns a list of up to top_k MatchResult objects.
        If no match, returns a single result with match_type='new'.
        """
        t0 = time.perf_counter()
        self._log_count += 1

        # Step 7.2
        content, norm = self._preprocess(raw_log)

        # Step 7.3
        vector = self._embed(norm)

        # Steps 7.4–7.5
        top = self._retrieve(vector)

        # Step 7.6
        if not top or top[0][1] < self.new_thr:
            return self._handle_new(norm, content, vector, t0)

        return self._handle_match(norm, top, vector, t0)

    def run_stream(
        self,
        online_csv:  str,
        output_dir:  str = "results",
        max_logs:    Optional[int] = None,
        verbose:     bool = False,
    ) -> Dict:
        """
        Step 12: Stream the full online CSV, evaluate, and save results.

        Parameters
        ----------
        online_csv  : path to {dataset}_online.csv
        output_dir  : where to save result CSV and summary JSON
        max_logs    : limit number of logs (None = all)
        verbose     : print each match result

        Returns
        -------
        dict with all online metrics (Step 15)
        """
        df = pd.read_csv(online_csv)
        if max_logs:
            df = df.head(max_logs)

        os.makedirs(output_dir, exist_ok=True)
        all_results = []
        counts = {"exact": 0, "partial": 0, "fuzzy": 0, "new": 0}

        print(f"\n[{self.dataset}] Streaming {len(df)} unseen logs ...")
        for _, row in df.iterrows():
            raw = str(row.get("raw", row.get("content", "")))
            results = self.match(raw)
            best    = results[0]
            all_results.append(best)
            counts[best.match_type] += 1

            if verbose:
                print(str(best))

            # Periodic entropy update (Module 3)
            if self._log_count % self.update_interval == 0:
                new_thr = self._ctrl.update()
                print(f"  [{self._log_count} logs]  store={len(self.store)}  "
                      f"new={counts['new']}  entropy_thr={new_thr:.3f}")

        # Save results
        df_out   = pd.DataFrame([asdict(r) for r in all_results])
        res_path = os.path.join(output_dir, f"{self.dataset}_stream_results.csv")
        df_out.to_csv(res_path, index=False)

        # Save updated store
        if self.auto_save:
            self.store.save()

        summary = self._compute_summary(all_results, counts, len(df))
        sum_path = os.path.join(output_dir,
                                f"{self.dataset}_stream_summary.json")
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)

        self._print_summary(summary)
        print(f"\n  Results : {res_path}")
        print(f"  Summary : {sum_path}")
        return summary

    # ── Steps 7.4–7.5: Cosine retrieval ──────────────────────────────

    def _retrieve(
        self, vector: np.ndarray
    ) -> List[Tuple[TemplateEntry, float]]:
        """
        Step 7.4 — cosine similarity with all stored centroids.
        Step 7.5 — sort and return top_k.
        """
        # Fast pre-filter: check keyword index first
        tokens   = vector  # placeholder — done inside store
        all_ents = self.store.all_entries()
        if not all_ents:
            return []

        centroids = np.stack([e._centroid_arr for e in all_ents])
        scores    = centroids @ vector   # (M,) cosine similarities

        top_idx = np.argsort(scores)[::-1][:self.top_k]
        return [(all_ents[i], float(scores[i])) for i in top_idx]

    # ── Step 7.6 + 8.1: Match found ───────────────────────────────────

    def _handle_match(
        self,
        norm:   str,
        top:    List[Tuple[TemplateEntry, float]],
        vector: np.ndarray,
        t0:     float,
    ) -> List[MatchResult]:
        best_entry, best_score = top[0]
        mtype = self._classify(best_score)

        # Step 8.1 — update centroid + frequency
        self.store.update_hit(best_entry.template_id, new_vector=vector)
        self._ctrl.record(int(best_entry.template_id, 16) % 10000)
        if self.auto_save:
            self.store.save()

        r = MatchResult(
            log_id        = self._log_count,
            normalised    = norm,
            match_type    = mtype,
            top1_template = best_entry.template,
            top1_score    = round(best_score, 4),
            store_size    = len(self.store),
            latency_ms    = round((time.perf_counter() - t0) * 1000, 2),
        )
        if len(top) >= 2:
            r.top2_template = top[1][0].template
            r.top2_score    = round(top[1][1], 4)
        if len(top) >= 3:
            r.top3_template = top[2][0].template
            r.top3_score    = round(top[2][1], 4)

        return [r]

    # ── Step 7.6 + 8.2: No match → new template ───────────────────────

    def _handle_new(
        self,
        norm:    str,
        content: str,
        vector:  np.ndarray,
        t0:      float,
    ) -> List[MatchResult]:
        new_tmpl = self._extract_template(norm, content)
        entry    = self.store.add(new_tmpl, vector, source="online")
        if self.auto_save:
            self.store.save()
        print(f"  [NEW] {new_tmpl[:70]}")

        return [MatchResult(
            log_id        = self._log_count,
            normalised    = norm,
            match_type    = "new",
            top1_template = new_tmpl,
            top1_score    = 1.0,
            store_size    = len(self.store),
            latency_ms    = round((time.perf_counter() - t0) * 1000, 2),
        )]

    def _classify(self, score: float) -> str:
        if score >= self.exact_thr:   return "exact"
        if score >= self.partial_thr: return "partial"
        return "fuzzy"

    # ── Preprocessing (Step 7.2) ──────────────────────────────────────

    def _preprocess(self, raw: str) -> Tuple[str, str]:
        content = raw.strip()
        if self._fmt_re:
            m = self._fmt_re.search(content)
            if m and "Content" in m.groupdict():
                content = m.group("Content").strip()
        norm = content
        for rex in self._rex:
            norm = rex.sub("<*>", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        return content, norm

    # ── Embedding (Step 7.3) ──────────────────────────────────────────

    def _embed(self, log: str) -> np.ndarray:
        v = self._sbert.encode(
            [log], convert_to_numpy=True, normalize_embeddings=True)[0]
        return v.astype(np.float32)

    # ── Template extraction for new logs ──────────────────────────────

    def _extract_template(self, norm: str, raw: str) -> str:
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
            elif re.match(r"^\d+$", tok) and len(tok) > 3:
                result.append("<*>")
            else:
                result.append(tok)
        cleaned, prev = [], False
        for tok in result:
            if tok == "<*>":
                if not prev: cleaned.append(tok)
                prev = True
            else:
                cleaned.append(tok); prev = False
        return " ".join(cleaned) if cleaned else "<*>"

    @staticmethod
    def _compile_format(log_format: str):
        headers, regex_str = [], ""
        for k, part in enumerate(re.split(r"(<[^<>]+>)", log_format)):
            if k % 2 == 0:
                regex_str += part.replace(" ", r"\s+")
            else:
                h = part.strip("<").strip(">")
                regex_str += f"(?P<{h}>.*?)"; headers.append(h)
        try:
            return re.compile("^" + regex_str + "$")
        except Exception:
            return None

    # ── Step 15: Metrics ──────────────────────────────────────────────

    def _compute_summary(self, results, counts, total) -> Dict:
        latencies = [r.latency_ms for r in results]
        scores    = [r.top1_score for r in results if r.match_type != "new"]
        matched   = counts["exact"] + counts["partial"] + counts["fuzzy"]
        return {
            "dataset":                  self.dataset,
            "total_logs_streamed":      total,
            "match_counts":             counts,
            "retrieval_accuracy":       round(matched / max(total, 1), 4),
            "exact_rate":               round(counts["exact"]   / max(total,1), 4),
            "partial_rate":             round(counts["partial"] / max(total,1), 4),
            "fuzzy_rate":               round(counts["fuzzy"]   / max(total,1), 4),
            "new_template_rate":        round(counts["new"]     / max(total,1), 4),
            "new_templates_discovered": counts["new"],
            "final_store_size":         len(self.store),
            "avg_latency_ms":           round(
                sum(latencies)/max(len(latencies),1), 3),
            "p95_latency_ms":           round(
                float(np.percentile(latencies, 95)), 3),
            "throughput_logs_per_sec":  round(
                1000 / max(sum(latencies)/max(len(latencies),1), 0.001), 1),
            "avg_top1_score":           round(
                sum(scores)/max(len(scores),1), 4),
        }

    def _print_summary(self, s: Dict):
        print(f"\n{'='*60}")
        print(f"  Online Results — {s['dataset']}")
        print(f"{'='*60}")
        print(f"  Logs streamed        : {s['total_logs_streamed']}")
        print(f"  Retrieval accuracy   : {s['retrieval_accuracy']:.4f}")
        print(f"  Exact  / Partial / Fuzzy: "
              f"{s['exact_rate']:.4f} / {s['partial_rate']:.4f} / {s['fuzzy_rate']:.4f}")
        print(f"  New templates found  : {s['new_templates_discovered']}")
        print(f"  Final store size     : {s['final_store_size']}")
        print(f"  Avg latency (ms)     : {s['avg_latency_ms']:.2f}")
        print(f"  P95 latency (ms)     : {s['p95_latency_ms']:.2f}")
        print(f"  Throughput (logs/s)  : {s['throughput_logs_per_sec']:.1f}")
        print(f"  Avg top-1 score      : {s['avg_top1_score']:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Online log matching — stream unseen logs, retrieve Top-3")
    parser.add_argument("--dataset",    type=str,   required=True,
                        help="Dataset name e.g. HDFS, BGL")
    parser.add_argument("--data_dir",   type=str,   default="datasets")
    parser.add_argument("--store_dir",  type=str,   default="repository")
    parser.add_argument("--output_dir", type=str,   default="results")
    parser.add_argument("--top_k",      type=int,   default=3)
    parser.add_argument("--new_thr",    type=float, default=0.60)
    parser.add_argument("--max_logs",   type=int,   default=None)
    parser.add_argument("--verbose",    action="store_true",
                        help="Print each match result")
    args = parser.parse_args()

    online_csv = os.path.join(
        args.data_dir, args.dataset, f"{args.dataset}_online.csv")
    if not os.path.exists(online_csv):
        print(f"Online CSV not found: {online_csv}")
        print(f"Run first: python preprocessing/preprocess.py --dataset {args.dataset}")
        sys.exit(1)

    matcher = LogMatcher(
        dataset   = args.dataset,
        store_dir = args.store_dir,
        top_k     = args.top_k,
        new_thr   = args.new_thr,
    )
    matcher.run_stream(
        online_csv  = online_csv,
        output_dir  = args.output_dir,
        max_logs    = args.max_logs,
        verbose     = args.verbose,
    )


if __name__ == "__main__":
    main()
