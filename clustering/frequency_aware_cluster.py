import numpy as np, re
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering

def first_keyword(log):
    for tok in log.split():
        if tok.startswith("<") and tok.endswith(">"): continue
        if re.match(r"^\d+$", tok): continue
        return tok.lower()
    return log[:20].lower()

def _score_freq(labels, n, frequencies):
    sizes = list(Counter(labels).values())
    max_frac = max(sizes)/n
    if max_frac > 0.60: return -1000.0
    sing = sum(1 for s in sizes if s==1)/len(sizes)
    label_to_freq = defaultdict(float)
    for lbl,freq in zip(labels,frequencies): label_to_freq[lbl]+=freq
    total_freq = sum(frequencies)
    freq_penalty = sum(label_to_freq[lbl]/total_freq for lbl,count in Counter(labels).items() if count==1)
    return np.log1p(len(sizes)) - sing*2.0 - max_frac*5.0 - freq_penalty*3.0

def two_stage_cluster_freq(embeddings, unique_logs, frequencies, dist, dataset="", n_clusters=None):
    n = len(unique_logs)
    global_labels = np.full(n,-1,dtype=int)
    next_label = 0
    freq_arr = np.array(frequencies, dtype=float)
    kw_groups = defaultdict(list)
    for i,log in enumerate(unique_logs):
        if dataset.lower()=="healthapp" and "|" in log:
            kw=log.split("|")[0].strip().lower()
        elif dataset.lower()=="openssh":
            toks=[t for t in log.split() if not (t.startswith("<") and t.endswith(">"))]
            kw=" ".join(toks[:2]).lower() if len(toks)>=2 else first_keyword(log)
        else: kw=first_keyword(log)
        kw_groups[kw].append(i)
    print(f"  [FreqAware] Stage 1: {len(kw_groups)} keyword groups")
    if n_clusters is not None and n_clusters>0:
        return AgglomerativeClustering(n_clusters=n_clusters,metric="euclidean",linkage="average").fit_predict(embeddings)
    for kw,indices in kw_groups.items():
        if len(indices)==1:
            global_labels[indices[0]]=next_label; next_label+=1; continue
        sub_emb=embeddings[indices]; sub_freqs=freq_arr[indices]
        best_lbl=None; best_score=-1e9
        for thr in [dist*0.5,dist*0.7,dist*0.85,dist,dist*1.1]:
            thr=max(0.05,min(0.99,thr))
            try:
                lbl=AgglomerativeClustering(n_clusters=None,distance_threshold=thr,metric="euclidean",linkage="average").fit_predict(sub_emb)
                score=_score_freq(lbl,len(indices),sub_freqs)
                if score>best_score: best_score=score; best_lbl=lbl
            except: continue
        if best_lbl is None: best_lbl=np.zeros(len(indices),dtype=int)
        offset_map={}
        for local,global_idx in zip(best_lbl,indices):
            if local not in offset_map: offset_map[local]=next_label; next_label+=1
            global_labels[global_idx]=offset_map[local]
    n_cl=len(set(global_labels)); sizes=Counter(global_labels.tolist())
    print(f"  [FreqAware] Stage 2: {n_cl} clusters  max={max(sizes.values())}  avg={np.mean(list(sizes.values())):.1f}")
    return global_labels
