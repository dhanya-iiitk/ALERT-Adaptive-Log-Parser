"""
template_extraction/contextual_variable_detector.py
==================================
Step 8: Extract log templates using BERT attention variance (Module 5).
High-variance token positions → <*>, Low-variance → constant.
"""

import re
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Optional


def extract_all_templates(
    unique_logs:    List[str],
    raw_contents:   List[str],
    labels:         np.ndarray,
    attentions:     Optional[List],
    tokenizer_name: str   = "bert-base-uncased",
    tau_r:          float = 0.50,
    variance_thr:   float = 0.01,
) -> Dict[int, str]:
    """
    Step 8: Extract one template per cluster.

    Parameters
    ----------
    unique_logs    : normalised log strings
    raw_contents   : original (unmasked) content strings
    labels         : cluster assignments (N,)
    attentions     : BERT attention tensors per log (or None)
    tokenizer_name : for subword alignment
    tau_r          : voting threshold
    variance_thr   : attention variance threshold

    Returns
    -------
    dict {cluster_id: template_string}
    """
    # Group logs by cluster
    cluster_logs = defaultdict(list)
    cluster_raw  = defaultdict(list)
    cluster_atts = defaultdict(list)

    for i, lbl in enumerate(labels):
        cluster_logs[lbl].append(unique_logs[i])
        cluster_raw[lbl].append(raw_contents[i])
        if attentions and i < len(attentions):
            cluster_atts[lbl].append(attentions[i])

    # Load tokenizer for attention alignment
    tokenizer = None
    if attentions:
        try:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            pass

    templates = {}
    for cid in cluster_logs:
        logs     = cluster_logs[cid]
        raws     = cluster_raw[cid]
        atts     = cluster_atts[cid] if cluster_atts else []

        if len(logs) == 1:
            tmpl = _template_single(logs[0], raws[0])
        elif atts and tokenizer:
            tmpl = _template_attention(logs, atts, tokenizer,
                                       tau_r=tau_r, variance_thr=variance_thr)
        else:
            tmpl = _template_voting(logs, tau_r=tau_r)

        templates[cid] = tmpl

    return templates


# ── Single log template ───────────────────────────────────────────────────

def _template_single(norm_log: str, raw_log: str) -> str:
    pre_toks = norm_log.split()
    raw_toks = raw_log.split()
    result   = []
    for i, tok in enumerate(pre_toks):
        if tok.startswith("<") and tok.endswith(">"):
            if i < len(raw_toks):
                m = re.match(r"^([A-Za-z_.]+)", raw_toks[i])
                if m and len(m.group(1)) >= 2:
                    result.append(m.group(1) + "<*>")
                    continue
            result.append("<*>")
        elif _is_var(tok):
            result.append("<*>")
        else:
            result.append(tok)
    return _clean(result)


# ── Attention variance method (Module 5 core) ──────────────────────────────

def _template_attention(logs, attentions, tokenizer,
                        tau_r=0.50, variance_thr=0.01) -> str:
    """
    Steps 5.1–5.4: Use BERT attention variance to identify variables.
    """
    token_attentions = []
    word_token_lists = []

    for log, att in zip(logs, attentions):
        if att is None:
            token_attentions.append([])
            word_token_lists.append(log.split())
            continue

        # att shape: (layers, heads, seq_len, seq_len)
        last_layer = att[-1]              # (heads, seq_len, seq_len)
        avg_heads  = last_layer.mean(0)   # (seq_len, seq_len)
        cls_row    = avg_heads[0, :]      # attention FROM [CLS]
        token_attentions.append(cls_row.numpy().tolist())

        enc   = tokenizer(log, add_special_tokens=True,
                          truncation=True, max_length=128)
        words = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        words = [w for w in words if w not in ("[CLS]", "[SEP]", "[PAD]")]
        word_token_lists.append(words)

    if not any(token_attentions):
        return _template_voting(logs, tau_r)

    max_len = max(len(w) for w in word_token_lists)
    result  = []

    for pos in range(max_len):
        pos_attentions = []
        pos_words      = []

        for wtoks, atts in zip(word_token_lists, token_attentions):
            if pos < len(wtoks):
                pos_words.append(wtoks[pos])
                att_idx = pos + 1
                if att_idx < len(atts):
                    pos_attentions.append(atts[att_idx])

        if not pos_words:
            continue

        if pos_attentions:
            variance    = float(np.var(pos_attentions))
            is_variable = variance > variance_thr
        else:
            is_variable = True

        if is_variable:
            result.append("<*>")
        else:
            mc, freq = Counter(pos_words).most_common(1)[0]
            result.append(mc if freq / len(pos_words) >= tau_r else "<*>")

    return _clean(result)


# ── Voting fallback ────────────────────────────────────────────────────────

def _template_voting(logs: List[str], tau_r: float = 0.50) -> str:
    seqs    = [lg.split() for lg in logs]
    max_len = max(len(s) for s in seqs)
    padded  = [s + [""] * (max_len - len(s)) for s in seqs]
    result  = []

    for col in range(max_len):
        col_toks = [padded[i][col] for i in range(len(padded)) if padded[i][col]]
        if not col_toks:
            result.append("<*>"); continue
        non_var = [t for t in col_toks if not _is_var(t)]
        if not non_var:
            result.append("<*>"); continue
        mc, freq = Counter(non_var).most_common(1)[0]
        result.append(mc if freq / len(col_toks) >= max(tau_r, 0.35) else "<*>")

    return _clean(result)


# ── Helpers ────────────────────────────────────────────────────────────────

def _is_var(token: str) -> bool:
    if token.startswith("<") and token.endswith(">"): return True
    if re.match(r"^\d+$", token) and len(token) > 3: return True
    if re.match(r"^0x[0-9a-fA-F]+$", token): return True
    if re.match(r"^(\d+\.){3}\d+$", token): return True
    if len(token) > 5 and sum(c.isdigit() for c in token) / len(token) > 0.6:
        return True
    return False


def _clean(tokens: List[str]) -> str:
    if not tokens: return "<*>"
    cleaned, prev = [], False
    for t in tokens:
        if t == "<*>":
            if not prev: cleaned.append(t)
            prev = True
        else:
            cleaned.append(t); prev = False
    return " ".join(cleaned) if cleaned else "<*>"
