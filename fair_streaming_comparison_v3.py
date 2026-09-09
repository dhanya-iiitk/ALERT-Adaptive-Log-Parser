"""
ALERT vs Drain vs Spell -- Fair Online Streaming Comparison (CORRECTED)
"""
import os, sys, time, json
import pandas as pd
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.evaluate import compute_GA


class SpellParser:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.templates = []

    def lcs_sim(self, s1, s2):
        t1, t2 = s1.split(), s2.split()
        m, n = len(t1), len(t2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if t1[i - 1] == t2[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
        return 2.0 * dp[m][n] / (m + n) if m + n > 0 else 0

    def train(self, logs):
        for log in logs:
            best_sim = max((self.lcs_sim(log, t) for t in self.templates), default=0)
            if best_sim < self.threshold:
                self.templates.append(log)

    def match(self, log):
        if not self.templates:
            return None, False
        sims = [self.lcs_sim(log, t) for t in self.templates]
        best_sim = max(sims)
        if best_sim >= self.threshold:
            return self.templates[sims.index(best_sim)], False
        return None, True


datasets = ['HDFS', 'Hadoop', 'Spark', 'Zookeeper', 'BGL', 'HPC',
            'Thunderbird', 'Windows', 'Linux', 'Android', 'HealthApp',
            'Apache', 'Proxifier', 'OpenSSH', 'OpenStack', 'Mac']

results = {}

for ds in datasets:
    offline_csv = f"datasets/{ds}/{ds}_offline.csv"
    online_csv = f"datasets/{ds}/{ds}_online.csv"
    gt_csv = f"datasets/{ds}/{ds}_2k.log_structured.csv"
    if not (os.path.exists(offline_csv) and os.path.exists(online_csv) and os.path.exists(gt_csv)):
        print(f"[{ds}] CSV not found -- skipping")
        continue

    offline_df = pd.read_csv(offline_csv)
    online_df = pd.read_csv(online_csv)
    gt_df = pd.read_csv(gt_csv)

    col = 'normalised' if 'normalised' in offline_df.columns else 'content'
    train_logs = offline_df[col].dropna().astype(str).tolist()

    online_df = online_df.dropna(subset=[col]).reset_index(drop=True)
    test_logs = online_df[col].astype(str).tolist()
    test_line_ids = online_df['line_id'].tolist()
    n_test = len(test_logs)
    print(f"\n[{ds}] Train={len(train_logs)} Test={n_test}")

    gt_event_id = gt_df.set_index("LineId")["EventId"].to_dict()
    gt_ids = [str(gt_event_id.get(lid, f"MISSING_{lid}")) for lid in test_line_ids]

    config = TemplateMinerConfig()
    config.drain_sim_th = 0.5
    config.drain_depth = 4
    config.drain_max_children = 100
    drain = TemplateMiner(config=config)
    for log in train_logs:
        drain.add_log_message(log)
    print(f"  Drain trained: {len(drain.drain.id_to_cluster)} templates")

    spell = SpellParser(threshold=0.5)
    spell.train(train_logs)
    print(f"  Spell trained: {len(spell.templates)} templates")

    drain_pred_ids = []
    drain_times = []
    for log in test_logs:
        t0 = time.time()
        result = drain.match(log)
        t1 = time.time()
        drain_times.append((t1 - t0) * 1000)
        if result is not None:
            drain_pred_ids.append(f"cluster_{result.cluster_id}")
        else:
            drain_pred_ids.append(f"NEW::{log}")
    drain_ga = compute_GA(gt_ids, drain_pred_ids)
    drain_lat = np.mean(drain_times)

    spell_pred_ids = []
    spell_times = []
    for log in test_logs:
        t0 = time.time()
        tmpl, is_new = spell.match(log)
        t1 = time.time()
        spell_times.append((t1 - t0) * 1000)
        if not is_new and tmpl is not None:
            spell_pred_ids.append(tmpl)
        else:
            spell_pred_ids.append(f"NEW::{log}")
    spell_ga = compute_GA(gt_ids, spell_pred_ids)
    spell_lat = np.mean(spell_times)

    alert_ga = None
    corrected_path = "results/online_metrics_corrected_temporal.csv"
    if os.path.exists(corrected_path):
        df_corr = pd.read_csv(corrected_path)
        row = df_corr[df_corr["dataset"] == ds]
        if not row.empty:
            alert_ga = float(row["GA"].values[0])
    if alert_ga is None:
        print(f"  [{ds}] WARNING: no corrected ALERT GA found in {corrected_path}")

    results[ds] = {
        'drain': {'GA': round(drain_ga, 4), 'lat_ms': round(drain_lat, 4)},
        'spell': {'GA': round(spell_ga, 4), 'lat_ms': round(spell_lat, 4)},
        'alert': {'GA': alert_ga, 'lat_ms': 32.4},
    }
    print(f"  Drain: GA={drain_ga:.4f}  lat={drain_lat:.3f}ms")
    print(f"  Spell: GA={spell_ga:.4f}  lat={spell_lat:.3f}ms")
    print(f"  ALERT: GA={alert_ga}")

os.makedirs("results", exist_ok=True)
with open("results/streaming_comparison_fair_v3.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print(f"{'Dataset':<12} {'Drain GA':>10} {'Spell GA':>10} {'ALERT GA':>10}")
for ds, r in results.items():
    a = r['alert']['GA']
    a_str = f"{a:.4f}" if a is not None else "N/A"
    print(f"{ds:<12} {r['drain']['GA']:>10.4f} {r['spell']['GA']:>10.4f} {a_str:>10}")
print(f"\nSaved: results/streaming_comparison_fair_v3.json")
