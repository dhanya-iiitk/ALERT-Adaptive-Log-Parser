"""
prepare_hdfs_v2_sample.py

Extracts the message content (stripping timestamp/level/logger prefix)
from a raw HDFS_v2 datanode log, takes a reproducible random sample,
and saves it persistently inside the project folder -- replacing the
old /tmp/HDFS_v2_normalised.txt which was lost because /tmp is not
persistent storage.
"""
import re
import random

RAW_LOG = "datasets/HDFS_v2_raw/hadoop-hdfs-datanode-mesos-32.log"
OUT_PATH = "datasets/HDFS_v2_normalised.txt"
SAMPLE_SIZE = 291
SEED = 42

LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+"
    r"\w+\s+[\w.$]+:\s*(.*)$"
)

contents = []
with open(RAW_LOG, "r", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        m = LINE_RE.match(line)
        if m:
            content = m.group(1).strip()
            if content:
                contents.append(content)

print(f"Extracted {len(contents)} valid content lines from raw log")

random.seed(SEED)
sample = random.sample(contents, min(SAMPLE_SIZE, len(contents)))

with open(OUT_PATH, "w") as f:
    for line in sample:
        f.write(line + "\n")

print(f"Saved {len(sample)}-line sample to {OUT_PATH} (seed={SEED}, reproducible)")
