"""
finetune_sbert_log.py
======================
Lightweight contrastive fine-tuning of SBERT
on log-domain data using ground-truth EventId pairs.

Datasets used: HDFS + OpenSSH + Proxifier
  - HDFS: well-structured, tests general improvement
  - OpenSSH: semantic conflation case (Accepted/Failed)
  - Proxifier: structural heterogeneity case

Positive pairs: logs with same EventId
Negative pairs: logs with different EventId

Run: python3 finetune_sbert_log.py
"""

import os, sys, time, random
import pandas as pd
import numpy as np
from collections import defaultdict

print("="*60)
print("SBERT LOG-DOMAIN CONTRASTIVE FINE-TUNING")
print("="*60)

# ── Step 1: Create contrastive pairs ─────────────────────────
print("\nStep 1: Creating contrastive pairs from ground truth...")

datasets_to_use = ['HDFS', 'OpenSSH', 'Proxifier']
data_dir = "datasets"

all_pairs = []  # (log1, log2, label) where label=1 positive, 0 negative
all_logs  = []  # all unique log contents

for ds in datasets_to_use:
    csv_path = f"{data_dir}/{ds}/{ds}_2k.log_structured.csv"
    if not os.path.exists(csv_path):
        print(f"  [{ds}] Not found — skipping")
        continue

    df = pd.read_csv(csv_path)
    df = df[['Content','EventId']].dropna()
    df['Content'] = df['Content'].astype(str).str.strip()

    # Group by EventId
    event_groups = defaultdict(list)
    for _, row in df.iterrows():
        event_groups[row['EventId']].append(row['Content'])

    event_ids = list(event_groups.keys())
    n_events  = len(event_ids)

    # Positive pairs: same EventId
    pos_pairs = []
    for eid, logs in event_groups.items():
        if len(logs) >= 2:
            sampled = random.sample(logs, min(len(logs), 10))
            for i in range(len(sampled)-1):
                pos_pairs.append((sampled[i], sampled[i+1], 1))

    # Negative pairs: different EventId
    neg_pairs = []
    eids = list(event_groups.keys())
    for i in range(min(len(pos_pairs), 200)):
        e1, e2 = random.sample(eids, 2)
        l1 = random.choice(event_groups[e1])
        l2 = random.choice(event_groups[e2])
        neg_pairs.append((l1, l2, 0))

    pairs = pos_pairs[:200] + neg_pairs[:200]
    random.shuffle(pairs)
    all_pairs.extend(pairs)

    print(f"  [{ds}] Events={n_events} "
          f"Pos={len(pos_pairs[:200])} "
          f"Neg={len(neg_pairs[:200])}")

print(f"\nTotal pairs: {len(all_pairs)}")
print(f"Positive: {sum(1 for _,_,l in all_pairs if l==1)}")
print(f"Negative: {sum(1 for _,_,l in all_pairs if l==0)}")

# ── Step 2: Load model ────────────────────────────────────────
print("\nStep 2: Loading all-MiniLM-L6-v2...")
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses
)
from torch.utils.data import DataLoader

t0 = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"  Model loaded in {time.time()-t0:.1f}s")

# ── Step 3: Baseline evaluation (generic SBERT) ──────────────
print("\nStep 3: Baseline evaluation (generic SBERT)...")

# OpenSSH Accepted vs Failed password test
test_pairs = [
    ("Accepted password for root from 192.168.1.1 port 22",
     "Failed password for root from 192.168.1.1 port 22"),
    ("Accepted password for admin from 10.0.0.1 port 22",
     "Failed password for admin from 10.0.0.1 port 22"),
    ("Accepted password for user1 from 172.16.0.1 port 22",
     "Failed password for user2 from 172.16.0.1 port 22"),
]

print("  OpenSSH semantic conflation test:")
print("  (Lower similarity = better discrimination)")
baseline_sims = []
for l1, l2 in test_pairs:
    emb = model.encode([l1, l2])
    sim = float(np.dot(emb[0], emb[1]) /
                (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))
    baseline_sims.append(sim)
    print(f"    Sim={sim:.4f}: '{l1[:30]}...' vs '{l2[:30]}...'")

print(f"  Mean similarity (generic): {np.mean(baseline_sims):.4f}")

# ── Step 4: Fine-tuning ───────────────────────────────────────
print("\nStep 4: Fine-tuning on log contrastive pairs...")
print("  (CPU fine-tuning — recording time...)")

train_examples = [
    InputExample(texts=[l1, l2], label=float(lbl))
    for l1, l2, lbl in all_pairs
]

train_dataloader = DataLoader(
    train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

t_start = time.time()
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=10,
    show_progress_bar=True,
)
t_end = time.time()
training_time = t_end - t_start

print(f"\n  Training completed!")
print(f"  Time on CPU: {training_time/60:.1f} minutes "
      f"({training_time:.0f} seconds)")
print(f"  Pairs: {len(all_pairs)}, Epochs: 2, Batch: 16")

# ── Step 5: Post fine-tuning evaluation ──────────────────────
print("\nStep 5: Post fine-tuning evaluation...")
print("  OpenSSH semantic conflation test:")
finetuned_sims = []
for l1, l2 in test_pairs:
    emb = model.encode([l1, l2])
    sim = float(np.dot(emb[0], emb[1]) /
                (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))
    finetuned_sims.append(sim)
    print(f"    Sim={sim:.4f}: '{l1[:30]}...' vs '{l2[:30]}...'")

print(f"  Mean similarity (fine-tuned): {np.mean(finetuned_sims):.4f}")
print(f"  Mean similarity (generic):    {np.mean(baseline_sims):.4f}")
improvement = np.mean(baseline_sims) - np.mean(finetuned_sims)
print(f"  Similarity reduction: {improvement:.4f} "
      f"({'improved' if improvement>0 else 'degraded'})")

# ── Step 6: Save fine-tuned model ────────────────────────────
print("\nStep 6: Saving fine-tuned model...")
model.save("models/sbert_log_finetuned")
print("  Saved to: models/sbert_log_finetuned/")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("FINE-TUNING SUMMARY")
print("="*60)
print(f"Training pairs:         {len(all_pairs)}")
print(f"Datasets used:          {', '.join(datasets_to_use)}")
print(f"Epochs:                 2")
print(f"Batch size:             16")
print(f"CPU training time:      {training_time/60:.1f} minutes")
print(f"Estimated GPU time:     ~{training_time/60/10:.1f} minutes")
print()
print("OpenSSH Accepted vs Failed similarity:")
print(f"  Generic SBERT:        {np.mean(baseline_sims):.4f}")
print(f"  Fine-tuned SBERT:     {np.mean(finetuned_sims):.4f}")
print(f"  Improvement:          {improvement:+.4f}")
