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
SBERT_MS = 27.0

summaries = {}
for ds in DATASETS:
    path = os.path.join(RESULTS_DIR, f"{ds}_stream_summary.json")
    with open(path) as f:
        summaries[ds] = json.load(f)

total_lat = [summaries[d]["avg_latency_ms"] for d in DATASETS]
store_size = [summaries[d]["final_store_size"] for d in DATASETS]

sbert_part = [min(SBERT_MS, t) for t in total_lat]
scoring_part = [max(0.0, t - SBERT_MS) for t in total_lat]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(DATASETS))

ax.bar(x, sbert_part, label=f"SBERT embedding (~{SBERT_MS:.0f}ms, constant)",
       color="#1f77b4")
ax.bar(x, scoring_part, bottom=sbert_part,
       label="Hybrid scoring + other (scales with repository size)",
       color="#ff7f0e")

ax.axhline(SBERT_MS, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(DATASETS, rotation=45, ha="right")
ax.set_ylabel("Per-log Latency (ms)")
ax.set_title("End-to-End Runtime Breakdown per Dataset\n"
             "SBERT Embedding (~27ms) is the Dominant Constant Bottleneck; "
             "Hybrid Scoring Scales with Repository Size")
ax.legend(loc="upper left")

max_i = int(np.argmax(total_lat))
min_i = int(np.argmin(total_lat))

# leave headroom above the tallest bar so the annotation never hits the title
ax.set_ylim(0, max(total_lat) * 1.28)

ax.annotate(f"{total_lat[max_i]:.1f}ms ({store_size[max_i]} templates)",
            xy=(max_i, total_lat[max_i]),
            xytext=(max_i - 4.0, total_lat[max_i] * 1.12),
            fontsize=9, color="darkred", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1))

ax.annotate(f"{total_lat[min_i]:.1f}ms ({store_size[min_i]} templates)",
            xy=(min_i, total_lat[min_i]),
            xytext=(min_i - 3.5, max(total_lat) * 0.35),
            fontsize=9, color="darkgreen", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "runtime_breakdown.png"), dpi=150)
plt.close()
print("Saved runtime_breakdown.png")

for i, d in enumerate(DATASETS):
    print(f"  {d:<12} total={total_lat[i]:>7.2f}ms  sbert={sbert_part[i]:>5.1f}ms  scoring={scoring_part[i]:>6.2f}ms  store={store_size[i]}")
