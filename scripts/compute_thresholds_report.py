#!/usr/bin/env python3
"""
Read models/embeddings_db.json (produced by compute_centroids_thresholds.py)
and write reports/thresholds_summary.json and thresholds_summary.csv
"""
import json, csv
from pathlib import Path
from statistics import mean, pstdev

IN = Path("models/embeddings_db.json")
OUT_DIR = Path("reports"); OUT_DIR.mkdir(exist_ok=True, parents=True)

if not IN.exists():
    print("ERROR: models/embeddings_db.json not found. Run compute_centroids_thresholds.py")
    raise SystemExit(1)

db = json.loads(IN.read_text())
rows = []
threshs = []
for k, v in sorted(db.items()):
    thr = float(v.get("threshold", 0.0))
    mean_d = float(v.get("mean_dist", 0.0))
    std_d = float(v.get("std_dist", 0.0))
    dim = len(v.get("centroid", []))
    rows.append({"registration": k, "centroid_dim": dim, "mean_dist": mean_d, "std_dist": std_d, "threshold": thr})
    threshs.append(thr)

summary = {
    "num_subjects": len(rows),
    "threshold_mean": mean(threshs) if threshs else 0.0,
    "threshold_std": pstdev(threshs) if len(threshs)>1 else 0.0,
    "threshold_min": min(threshs) if threshs else 0.0,
    "threshold_max": max(threshs) if threshs else 0.0
}

(OUT_DIR / "thresholds_summary.json").write_text(json.dumps({"summary": summary, "per_subject": rows}, indent=2))
with open(OUT_DIR / "thresholds_summary.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["registration","centroid_dim","mean_dist","std_dist","threshold"])
    w.writeheader(); w.writerows(rows)

print("Wrote reports/thresholds_summary.json and .csv")
print("Summary:", summary)
