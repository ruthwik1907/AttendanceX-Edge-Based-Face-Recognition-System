#!/usr/bin/env python3
"""
Count images per student in data/processed/, compute total images, and
write reports/dataset_stats.json and reports/dataset_stats.csv
"""
import json, csv
from pathlib import Path

DATA = Path("data/processed")
OUT_DIR = Path("reports"); OUT_DIR.mkdir(exist_ok=True, parents=True)

if not DATA.exists():
    print("ERROR: data/processed/ not found. Run process_all.py first. See scripts/process_all.py for details.")
    raise SystemExit(1)

rows = []
total = 0
for reg in sorted([d for d in DATA.iterdir() if d.is_dir()]):
    imgs = [p for p in reg.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")]
    n = len(imgs)
    rows.append({"registration": reg.name, "num_images": n})
    total += n

summary = {"num_subjects": len(rows), "total_images": total}
(OUT_DIR / "dataset_stats.json").write_text(json.dumps({"summary": summary, "per_subject": rows}, indent=2))

with open(OUT_DIR / "dataset_stats.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["registration","num_images"])
    w.writeheader()
    w.writerows(rows)

print("Wrote reports/dataset_stats.json and .csv")
print("Subjects:", summary["num_subjects"], "Total images:", summary["total_images"])
