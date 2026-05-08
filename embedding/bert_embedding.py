"""
embedding/bert_embedding.py
============================
Step 6: Generate SBERT embeddings for unique log messages.

Uses sentence-transformers (SBERT) instead of raw BERT because:
- Designed specifically for semantic similarity tasks
- No manual mean-pooling needed
- Better clustering results
- Faster inference

Model used: all-MiniLM-L6-v2 (fast, strong performance)
Alternative: all-mpnet-base-v2 (slower, slightly better)
"""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


def generate_embeddings(
    logs:              List[str],
    bert_model:        str  = "all-MiniLM-L6-v2",
    max_length:        int  = 128,
    batch_size:        int  = 64,
    return_attentions: bool = False,
) -> Tuple[np.ndarray, Optional[List]]:
    """
    Generate SBERT embeddings for a list of log strings.

    Parameters
    ----------
    logs              : list of normalised log strings
    bert_model        : SBERT model name
                        'all-MiniLM-L6-v2'  — fast, good quality
                        'all-mpnet-base-v2' — slower, best quality
    max_length        : max token length (ignored by SBERT, kept for compat)
    batch_size        : inference batch size
    return_attentions : kept for API compatibility (SBERT has no attentions)

    Returns
    -------
    embeddings  : np.ndarray  (N, embedding_dim)  L2-normalised
    attentions  : None  (SBERT does not expose attention weights)
    """
    if not SBERT_AVAILABLE:
        raise ImportError(
            "sentence-transformers not installed.\n"
            "Run: pip3 install sentence-transformers")

    print(f"[Embedding] Loading SBERT model: {bert_model}")
    model = SentenceTransformer(bert_model)
    print(f"[Embedding] Encoding {len(logs)} logs ...")

    embeddings = model.encode(
        logs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise built-in
    )

    print(f"[Embedding] Done. Shape: {embeddings.shape}")

    # return_attentions=False always for SBERT
    return embeddings, None
