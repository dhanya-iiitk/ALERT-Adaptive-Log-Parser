import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["HDFS", "Hadoop", "Spark", "Zookeeper", "BGL", "HPC",
            "Thunderbird", "Windows", "Linux", "Android", "HealthApp",
            "Apache", "Proxifier", "OpenSSH", "OpenStack", "Mac"]

RESULTS_DIR = "results"
OUT_DIR = "."

summaries = {}
for ds in DATASETS:
    path = os.path.join(RESULTS_DIR, f"{ds}_stream_summary.json")
    with open(path) as f:
        summaries[ds] = json.load(f)

exact = [summaries[d]["exact_rate"] * 100 for d in DATASETS]
partial = [summaries[d]["partial_rate"] * 100 for d in DATASETS]
fuzzy = [summaries[d]["fuzzy_rate"] * 100 for d in DATASETS]
new = [summaries[d]["new_template_rate"] * 100 for d in DATASETS]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(DATASETS))

ax.bar(x, exact, label="Exact", color="#2ca02c")
ax.bar(x, partial, bottom=exact, label="Partial", color="#1f77b4")
bottom2 = [e + p for e, p in zip(exact, partial)]
ax.bar(x, fuzzy, bottom=bottom2, label="Fuzzy", color="#ff7f0e")
bottom3 = [b + f for b, f in zip(bottom2, fuzzy)]
ax.bar(x, new, bottom=bottom3, label="New", color="#d62728")

ax.set_xticks(x)
ax.set_xticklabels(DATASETS, rotation=45, ha="right")
ax.set_ylabel("match type (%)")
ax.set_ylim(0, 100)
ax.legend(loc="upper right", ncol=4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "match_type_distribution.png"), dpi=150)
plt.close()
print("Saved match_type_distribution.png")

for d in DATASETS:
    i = DATASETS.index(d)
    print(f"  {d:<12} exact={exact[i]:.1f}%  partial={partial[i]:.1f}%  fuzzy={fuzzy[i]:.1f}%  new={new[i]:.1f}%")

scores = [summaries[d]["avg_top1_score"] for d in DATASETS]

def bar_color(s):
    if s >= 0.99:
        return "#2ca02c"
    elif s >= 0.97:
        return "#1f77b4"
    elif s >= 0.95:
        return "#ff7f0e"
    else:
        return "#d62728"

colors = [bar_color(s) for s in scores]

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x, scores, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(DATASETS, rotation=45, ha="right")
ax.set_ylabel("avg similarity score")
ymin = max(0.0, min(scores) - 0.03)
ax.set_ylim(ymin, 1.01)

from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#2ca02c", label="score >= 0.99 (very high)"),
    Patch(facecolor="#1f77b4", label="score >= 0.97 (high)"),
    Patch(facecolor="#ff7f0e", label="score >= 0.95 (moderate)"),
    Patch(facecolor="#d62728", label="score < 0.95 (lower)"),
]
ax.legend(handles=legend_elems, loc="lower left", ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "average_similarity_score.png"), dpi=150)
plt.close()
print("Saved average_similarity_score.png")

for d in DATASETS:
    print(f"  {d:<12} avg_top1_score={summaries[d]['avg_top1_score']:.4f}")
