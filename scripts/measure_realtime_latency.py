#!/usr/bin/env python3
"""
Measure inference+classification latency by running the same preprocess->emb->classify
steps as realtime_tflite_infer.py but on images in data/processed (no camera).
Outputs reports/latency_report.json and latency_samples.csv
"""
import time, json, csv, numpy as np, sys
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

MODELS = Path("models")
DATA = Path("data/processed")
OUT = Path("reports"); OUT.mkdir(exist_ok=True)

EMB = MODELS / "mobilefacenet.tflite"
CLS = MODELS / "classifier.tflite"
if not EMB.exists() or not CLS.exists():
    print("ERROR: Required TFLite models missing in models/.")
    sys.exit(1)

# load interpreters
emb_i = tflite.Interpreter(model_path=str(EMB)); emb_i.allocate_tensors()
emb_in = emb_i.get_input_details()[0]; emb_out = emb_i.get_output_details()[0]

cls_i = tflite.Interpreter(model_path=str(CLS)); cls_i.allocate_tensors()
cls_in = cls_i.get_input_details()[0]; cls_out = cls_i.get_output_details()[0]

# collect sample image paths (one per subject)
samples = []
for d in DATA.iterdir():
    if not d.is_dir(): continue
    found = list(d.glob("*.jpg"))
    if found:
        samples.append(str(found[0]))
if not samples:
    print("No sample images found under data/processed/")
    sys.exit(1)

import cv2
def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError("Can't read " + img_path)
    H = emb_in["shape"][1]; W = emb_in["shape"][2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H)).astype("float32")
    float_input = img / 127.5 - 1.0
    if emb_in["dtype"] == np.float32:
        return float_input.astype("float32").reshape(1,H,W,3)
    else:
        scale, zp = emb_in.get("quantization", (0.0,0))
        if scale != 0:
            q = float_input / scale + zp
            if np.issubdtype(emb_in["dtype"], np.signedinteger):
                q = q.round().astype(np.int8)
            else:
                q = q.round().astype(np.uint8)
            return q.reshape(1,H,W,3)
        else:
            return float_input.astype("float32").reshape(1,H,W,3)

N = min(200, len(samples)*3)
latencies = []
samples_run = 0
for i in range(N):
    path = samples[i % len(samples)]
    inp = preprocess(path)
    t0 = time.perf_counter()
    emb_i.set_tensor(emb_in["index"], inp)
    emb_i.invoke()
    emb_raw = emb_i.get_tensor(emb_out["index"])
    emb = emb_raw.astype("float32").reshape(1,-1)
    # classifier
    # handle quantization as in realtime script
    if cls_in["dtype"] == np.float32:
        cls_inp = emb.astype("float32")
    else:
        scale, zp = cls_in.get("quantization", (0.0,0))
        if scale != 0:
            q = emb / scale + zp
            if np.issubdtype(cls_in["dtype"], np.signedinteger):
                cls_inp = q.round().astype(np.int8)
            else:
                cls_inp = q.round().astype(np.uint8)
        else:
            cls_inp = emb.astype(cls_in["dtype"])
    cls_i.set_tensor(cls_in["index"], cls_inp)
    cls_i.invoke()
    _ = cls_i.get_tensor(cls_out["index"])
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000.0)  # ms
    samples_run += 1

import statistics
report = {
    "samples": samples_run,
    "mean_ms": statistics.mean(latencies),
    "median_ms": statistics.median(latencies),
    "stdev_ms": statistics.pstdev(latencies) if len(latencies)>1 else 0.0,
    "min_ms": min(latencies),
    "max_ms": max(latencies)
}
(OUT / "latency_report.json").write_text(json.dumps(report, indent=2))
with open(OUT / "latency_samples.csv","w",newline="",encoding="utf-8") as f:
    import csv
    w = csv.writer(f)
    w.writerow(["index","ms"])
    for i, v in enumerate(latencies):
        w.writerow([i, float(v)])
print("Wrote reports/latency_report.json and latency_samples.csv")
print(report)
