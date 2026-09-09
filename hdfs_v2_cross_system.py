"""
HDFS_v2 Cross-System Evaluation
Tests ALERT's generalisation to HDFS_v2 logs
using stores built from 16 LogHub-2k datasets
"""

import os, sys, json, time
import numpy as np

sys.path.insert(0, '.')

print("Loading SBERT...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load HDFS_v2 logs
with open('datasets/HDFS_v2_normalised.txt') as f:
    logs = [l.strip() for l in f if l.strip()]
print(f"Loaded {len(logs)} HDFS_v2 logs")

# Embed
print("Embedding...")
embeddings = model.encode(logs, batch_size=64,
                          show_progress_bar=True,
                          normalize_embeddings=True)

def load_store(path):
    with open(path) as f:
        data = json.load(f)
    for e in data:
        e['_centroid'] = np.array(e['centroid'],
                                   dtype=np.float32)
        toks = e['template'].split()
        e['_fixed'] = [t for t in toks if t != '<*>']
        e['_first_kw'] = toks[0].lower() if toks else ''
    return data

def structural_score(log, template):
    log_toks  = set(log.lower().split())
    fixed     = set(f.lower() for f in template['_fixed'])
    if not fixed: return 0.5
    overlap   = len(log_toks & fixed) / len(fixed)
    kw_match  = 1.0 if (log.split()[0].lower() if log.split()
                        else '') == template['_first_kw'] else 0.0
    log_len   = len(log.split())
    tmpl_len  = len(template['template'].split())
    len_sim   = 1.0 - abs(log_len - tmpl_len) / max(log_len,
                                                      tmpl_len, 1)
    return 0.5*overlap + 0.3*kw_match + 0.2*len_sim

def hybrid_score(emb, log, template, alpha=0.7):
    cos = float(np.dot(emb, template['_centroid']))
    st  = structural_score(log, template)
    return alpha*cos + (1-alpha)*st

datasets = ['HDFS','Hadoop','Spark','Zookeeper','BGL','HPC',
            'Thunderbird','Windows','Linux','Android','HealthApp',
            'Apache','Proxifier','OpenSSH','OpenStack','Mac']

TAU_NEW = 0.60
results = {}

for ds in datasets:
    store_path = f'repository/{ds}_template_store.json'
    if not os.path.exists(store_path):
        print(f"[{ds}] store not found")
        continue

    store   = load_store(store_path)
    matched = 0
    new     = 0
    scores  = []

    exact = partial = fuzzy = 0
    for log, emb in zip(logs, embeddings):
        first_kw   = log.split()[0].lower() if log.split() else ''
        candidates = [t for t in store
                      if t['_first_kw'] == first_kw]
        if not candidates:
            candidates = store

        best = max(hybrid_score(emb, log, t, alpha=0.7)
                   for t in candidates)
        scores.append(best)
        if best >= 0.99:
            exact += 1; matched += 1
        elif best >= 0.80:
            partial += 1; matched += 1
        elif best >= TAU_NEW:
            fuzzy += 1; matched += 1
        else:
            new += 1

    results[ds] = {
        'matched':    matched,
        'exact':      exact,
        'partial':    partial,
        'fuzzy':      fuzzy,
        'new':        new,
        'total':      len(logs),
        'match_rate': matched/len(logs),
        'exact_rate': exact/len(logs),
        'partial_rate': partial/len(logs),
        'fuzzy_rate': fuzzy/len(logs),
        'new_rate':   new/len(logs),
        'mean_score': float(np.mean(scores)),
    }
    print(f"  [{ds:<12}] matched={matched:>3} "
          f"new={new:>3} "
          f"match={matched/len(logs):.3f} "
          f"score={np.mean(scores):.4f}")

print()
print("="*60)
print("HDFS_v2 CROSS-SYSTEM RESULTS")
print("="*60)
print(f"{'Dataset':<14} {'Match%':>8} {'New%':>8} "
      f"{'MeanScore':>10}")
print("-"*42)
for ds, r in results.items():
    print(f"{ds:<14} {r['match_rate']:>8.4f} "
          f"{r['new_rate']:>8.4f} "
          f"{r['mean_score']:>10.4f}")

best = max(results, key=lambda x: results[x]['match_rate'])
hdfs_new = results.get('HDFS', {}).get('new_rate', 0)
total_new_hdfs = results.get('HDFS', {}).get('new', 0)

print()
print(f"Best matching store:  {best} "
      f"({results[best]['match_rate']:.4f})")
print(f"HDFS store new rate:  {hdfs_new:.4f} "
      f"({total_new_hdfs} new templates)")
print(f"Total logs tested:    {len(logs)}")

os.makedirs('results', exist_ok=True)
with open('results/hdfs_v2_cross_system.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/hdfs_v2_cross_system.json")
