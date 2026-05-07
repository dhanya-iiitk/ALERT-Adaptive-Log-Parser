"""
embedding/bert_embedding.py
======================
Step 6: Generate BERT embeddings for unique log messages.
Returns L2-normalised mean-pool vectors + optional attention weights.
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    from transformers import BertTokenizer, BertModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


def generate_embeddings(
    logs:              List[str],
    bert_model:        str  = "bert-base-uncased",
    max_length:        int  = 128,
    batch_size:        int  = 32,
    return_attentions: bool = False,
) -> Tuple[np.ndarray, Optional[List]]:
    """
    Generate BERT embeddings for a list of log strings.

    Parameters
    ----------
    logs              : list of normalised log strings
    bert_model        : HuggingFace model name
    max_length        : max BERT token length
    batch_size        : inference batch size
    return_attentions : return per-log attention tensors

    Returns
    -------
    embeddings  : np.ndarray  (N, 768)  L2-normalised
    attentions  : list of tensors or None
    """
    if not BERT_AVAILABLE:
        raise ImportError("torch and transformers are required.")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model     = BertModel.from_pretrained(bert_model, output_attentions=return_attentions)
    model.to(device)
    model.eval()
    print(f"[Embedding] BERT loaded on {device} | {len(logs)} logs")

    n          = len(logs)
    embeddings = np.zeros((n, 768), dtype=np.float32)
    all_atts   = [] if return_attentions else None

    from tqdm import tqdm
    for i in tqdm(range(0, n, batch_size), desc="Encoding"):
        batch = logs[i: i + batch_size]
        enc   = tokenizer.batch_encode_plus(
            list(batch), add_special_tokens=True,
            max_length=max_length, padding="max_length",
            truncation=True, return_attention_mask=True, return_tensors="pt")
        ids   = enc["input_ids"].to(device)
        mask  = enc["attention_mask"].to(device)

        with torch.no_grad():
            out  = model(input_ids=ids, attention_mask=mask)
            toks = out.last_hidden_state
            mexp = mask.unsqueeze(-1).float()
            emb  = (toks * mexp).sum(1) / mexp.sum(1).clamp(min=1e-9)

        embeddings[i: i + len(batch)] = emb.cpu().numpy()

        if return_attentions and out.attentions:
            stacked = torch.stack(out.attentions, dim=0)
            for b in range(len(batch)):
                all_atts.append(stacked[:, b, :, :, :].cpu())

    # L2 normalise
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    print(f"[Embedding] Done. Shape: {embeddings.shape}")
    return embeddings, all_atts
