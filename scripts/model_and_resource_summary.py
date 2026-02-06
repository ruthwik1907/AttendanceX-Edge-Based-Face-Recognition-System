#!/usr/bin/env python3
"""
Report model sizes in bytes and, if psutil available, run one inference and sample memory/CPU.
Writes reports/resource_summary.json
"""
import json, os, time
from pathlib import Path

MODELS = Path("models")
OUT = Path("reports"); OUT.mkdir(exist_ok=True, parents=True)

files = {}
for p in MODELS.glob("*"):
    if p.is_file():
        files[p.name] = {"size_bytes": p.stat().st_size}

# optional live sampling
try:
    import psutil, numpy as np
    has_psutil = True
except Exception:
    has_psutil = False

sample = {"has_psutil": has_psutil}

if has_psutil:
    # attempt a small inference using mobilefacenet.tflite (if present)
    import tflite_runtime.interpreter as tflite
    emb = MODELS / "mobilefacenet.tflite"
    cls = MODELS / "classifier.tflite"
    if emb.exists() and cls.exists():
        interp_e = tflite.Interpreter(model_path=str(emb)); interp_e.allocate_tensors()
        interp_c = tflite.Interpreter(model_path=str(cls)); interp_c.allocate_tensors()
        # create a random input with correct shape
        in_e = interp_e.get_input_details()[0]
        import numpy as np
        shape = in_e["shape"]
        dummy = np.zeros(shape, dtype=np.float32)
        p = psutil.Process()
        mem_before = p.memory_info().rss
        cpu_before = p.cpu_percent(interval=None)
        t0 = time.perf_counter()
        interp_e.set_tensor(in_e["index"], dummy)
        interp_e.invoke()
        emb_out = interp_e.get_tensor(interp_e.get_output_details()[0]["index"])
        # feed to classifier if shape matches
        try:
            in_c = interp_c.get_input_details()[0]
            # flatten as needed
            cinput = emb_out.astype("float32").reshape(1, -1)
            if in_c["dtype"] != cinput.dtype:
                cinput = cinput.astype(in_c["dtype"])
            interp_c.set_tensor(in_c["index"], cinput)
            interp_c.invoke()
            _ = interp_c.get_tensor(interp_c.get_output_details()[0]["index"])
        except Exception as e:
            pass
        t1 = time.perf_counter()
        mem_after = p.memory_info().rss
        cpu_after = p.cpu_percent(interval=None)
        sample.update({
            "sample_latency_ms": (t1 - t0) * 1000.0,
            "memory_rss_before": mem_before,
            "memory_rss_after": mem_after,
            "cpu_percent": cpu_after
        })

out = {"models": files, "sample": sample}
(OUT / "resource_summary.json").write_text(json.dumps(out, indent=2))
print("Wrote reports/resource_summary.json")
print(out)
