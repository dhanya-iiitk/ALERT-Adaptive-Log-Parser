"""
repository/template_store.py
==============================
Module 6 — Template Repository Construction Module

Persistent semantic template memory with centroid vectors,
frequency counts, and timestamps. Each dataset gets its own
independent store file: HDFS_template_store.json, BGL_template_store.json, etc.

Steps
-----
6.1  Template ID assignment  (MD5 hash of template string)
6.2  Centroid computation    (mean of cluster embeddings)
6.3  Repository storage      (JSON file per dataset)
"""

import json
import os
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _make_id(template: str) -> str:
    """Step 6.1 — unique template ID from MD5 hash."""
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# TemplateEntry — one record in the repository
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemplateEntry:
    template_id:  str            # Step 6.1
    template:     str            # Template string with <*> wildcards
    centroid:     List[float]    # Step 6.2 — mean embedding vector (768-d)
    occurrences:  int  = 1
    first_seen:   str  = ""
    last_seen:    str  = ""
    source:       str  = "parser"   # 'parser' or 'online'
    dataset:      str  = ""

    # Runtime only — not persisted to JSON
    _centroid_arr: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False)
    _fixed_tokens: List[str] = field(
        default_factory=list, repr=False, compare=False)
    _first_kw:     str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        self._centroid_arr = np.array(self.centroid, dtype=np.float32)
        self._build_index()

    def _build_index(self):
        toks = self.template.split()
        self._fixed_tokens = [t for t in toks if t != "<*>"]
        self._first_kw = self._fixed_tokens[0].lower() if self._fixed_tokens else ""

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "template":    self.template,
            "centroid":    self.centroid,
            "occurrences": self.occurrences,
            "first_seen":  self.first_seen,
            "last_seen":   self.last_seen,
            "source":      self.source,
            "dataset":     self.dataset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemplateEntry":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    def update_centroid(self, new_vector: np.ndarray):
        """
        Step 8.1 — Rolling centroid update using exponential moving average.
        centroid_new = α * new_vector + (1-α) * centroid_old
        α decreases as occurrences grow → stable over time.
        """
        alpha = 1.0 / (self.occurrences + 1)
        updated = alpha * new_vector + (1 - alpha) * self._centroid_arr
        norm = np.linalg.norm(updated)
        self._centroid_arr = updated / (norm + 1e-8)
        self.centroid = self._centroid_arr.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# TemplateStore — the per-dataset repository
# ─────────────────────────────────────────────────────────────────────────────

class TemplateStore:
    """
    Module 6: Per-dataset Template Repository.

    One instance per dataset. Never mix templates across datasets.

    Parameters
    ----------
    dataset          : dataset name (e.g. 'HDFS')
    store_dir        : directory for JSON files (default 'repository')
    max_templates    : soft cap before compaction (Step 8.3)
    compact_sim_thr  : cosine similarity above which two templates are merged
    """

    def __init__(
        self,
        dataset:         str,
        store_dir:       str   = "repository",
        max_templates:   int   = 5000,
        compact_sim_thr: float = 0.98,
    ):
        self.dataset         = dataset
        self.max_templates   = max_templates
        self.compact_sim_thr = compact_sim_thr
        self.store_path      = os.path.join(
            store_dir, f"{dataset}_template_store.json")

        self._entries:   Dict[str, TemplateEntry] = {}
        self._kw_index:  Dict[str, List[str]]     = {}  # first_kw -> [ids]

        os.makedirs(store_dir, exist_ok=True)
        if os.path.exists(self.store_path):
            self.load()

    # ── Load / Save ──────────────────────────────────────────────────

    def load(self):
        """Step 6.3 — Load from {dataset}_template_store.json."""
        with open(self.store_path) as f:
            data = json.load(f)
        self._entries = {}
        self._kw_index = {}
        for d in data:
            e = TemplateEntry.from_dict(d)
            self._register(e)
        print(f"[{self.dataset} Store] Loaded {len(self._entries)} templates")

    def save(self):
        """Step 6.3 — Persist to {dataset}_template_store.json."""
        with open(self.store_path, "w") as f:
            json.dump([e.to_dict() for e in self._entries.values()],
                      f, indent=2)

    # ── Add / Update ──────────────────────────────────────────────────

    def add(
        self,
        template: str,
        centroid: np.ndarray,
        source:   str = "parser",
        count:    int = 1,
    ) -> TemplateEntry:
        """Step 6.1–6.3 — Add template; update centroid if duplicate."""
        tid = _make_id(template)
        if tid in self._entries:
            e = self._entries[tid]
            e.occurrences += count
            e.last_seen    = _now()
            e.update_centroid(centroid)
            return e

        now   = _now()
        entry = TemplateEntry(
            template_id = tid,
            template    = template,
            centroid    = centroid.tolist(),
            occurrences = count,
            first_seen  = now,
            last_seen   = now,
            source      = source,
            dataset     = self.dataset,
        )
        self._register(entry)

        # Step 8.3 — compact if over capacity
        if len(self._entries) > self.max_templates:
            self._compact()

        return entry

    def update_hit(self, template_id: str, new_vector: Optional[np.ndarray] = None):
        """Step 8.1 — Increment frequency, update centroid (EMA)."""
        if template_id not in self._entries:
            return
        e = self._entries[template_id]
        e.occurrences += 1
        e.last_seen    = _now()
        if new_vector is not None:
            e.update_centroid(new_vector)

    # ── Query ─────────────────────────────────────────────────────────

    def get_by_keyword(self, keyword: str) -> List[TemplateEntry]:
        ids = self._kw_index.get(keyword.lower(), [])
        return [self._entries[i] for i in ids if i in self._entries]

    def all_entries(self) -> List[TemplateEntry]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    # ── Step 8.3: Compaction ──────────────────────────────────────────

    def _compact(self):
        """Merge highly similar templates to keep repository bounded."""
        print(f"[{self.dataset} Store] Compaction: "
              f"{len(self._entries)} > {self.max_templates}")
        entries   = list(self._entries.values())
        centroids = np.stack([e._centroid_arr for e in entries])
        sim       = centroids @ centroids.T   # cosine similarity matrix

        merged = set()
        for i in range(len(entries)):
            if i in merged: continue
            for j in range(i + 1, len(entries)):
                if j in merged: continue
                if sim[i, j] >= self.compact_sim_thr:
                    ei, ej = entries[i], entries[j]
                    victim = ej if ei.occurrences >= ej.occurrences else ei
                    surv   = ei if victim is ej else ej
                    surv.occurrences += victim.occurrences
                    surv.update_centroid(victim._centroid_arr)
                    del self._entries[victim.template_id]
                    kw = victim._first_kw
                    if kw in self._kw_index:
                        self._kw_index[kw] = [
                            x for x in self._kw_index[kw]
                            if x != victim.template_id]
                    merged.add(j)

        print(f"[{self.dataset} Store] After compaction: "
              f"{len(self._entries)} templates (merged {len(merged)})")

    # ── Internal ──────────────────────────────────────────────────────

    def _register(self, entry: TemplateEntry):
        self._entries[entry.template_id] = entry
        kw = entry._first_kw
        if kw not in self._kw_index:
            self._kw_index[kw] = []
        if entry.template_id not in self._kw_index[kw]:
            self._kw_index[kw].append(entry.template_id)
