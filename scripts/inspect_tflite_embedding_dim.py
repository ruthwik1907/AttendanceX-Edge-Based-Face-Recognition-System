#!/usr/bin/env python3
"""
Inspect TFLite embedding model to get exact output dimension and input shape.
Writes models/embedding_model_info.json
"""
import json, sys
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

MODELS = Path("models")
EMB = MODELS / "mobilefacenet.tflite"
OUT = MODELS / "embedding_model_info.json"

if not EMB.exists():
    print("ERROR: models/mobilefacenet.tflite not found.")
    sys.exit(1)

interp = tflite.Interpreter(model_path=str(EMB))
interp.allocate_tensors()
ins = interp.get_input_details()
outs = interp.get_output_details()

info = {
    "input_details": [],
    "output_details": []
}
for i in ins:
    info["input_details"].append({
        "shape": list(i.get("shape", [])),
        "dtype": str(i.get("dtype")),
        "quantization": i.get("quantization", None)
    })
for o in outs:
    info["output_details"].append({
        "shape": list(o.get("shape", [])),
        "dtype": str(o.get("dtype")),
        "quantization": o.get("quantization", None)
    })

OUT.write_text(json.dumps(info, indent=2))
print("Wrote", OUT)
print("Embedding output shape:", info["output_details"])
