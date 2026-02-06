#!/usr/bin/env python3
"""
Gather all reports into a single JSON for paper insertion.
"""
import json
from pathlib import Path

OUT = Path("reports"); OUT.mkdir(exist_ok=True)
files = [
    "dataset_stats.json",
    "thresholds_summary.json",
    "classifier_eval.json",
    "latency_report.json",
    "resource_summary.json"
]
bundle = {}
for fn in files:
    p = OUT / fn
    if p.exists():
        bundle[fn] = json.loads(p.read_text())
    else:
        bundle[fn] = {"missing": True}

(OUT / "paper_metrics_bundle.json").write_text(json.dumps(bundle, indent=2))
print("Wrote reports/paper_metrics_bundle.json — use this to fill placeholders in the draft.")
