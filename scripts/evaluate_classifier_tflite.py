#!/usr/bin/env python3
"""
Evaluate models/classifier.tflite on saved train_embeddings.npz (from scripts/train_classifier_from_tflite_embeddings.py).
Outputs reports/classifier_eval.json and confusion_matrix.csv
"""
import json, numpy as np, csv, sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

MODELS = Path("models")
DATA = MODELS / "train_embeddings.npz"
CLS = MODELS / "classifier.tflite"
LABEL_MAP = MODELS / "label_map.json"
OUT = Path("reports"); OUT.mkdir(exist_ok=True)

if not DATA.exists():
    print("ERROR: models/train_embeddings.npz missing. Run scripts/train_classifier_from_tflite_embeddings.py first.")
    sys.exit(1)
if not CLS.exists():
    print("ERROR: models/classifier.tflite missing.")
    sys.exit(1)

d = np.load(DATA, allow_pickle=True)
X = d["X"]
y = d["y"]

# label inversion from label_map (index -> registration)
label_map = json.loads(LABEL_MAP.read_text())
inv_map = {int(k): v for k, v in label_map.items()}

# load interpreter
interp = tflite.Interpreter(model_path=str(CLS)); interp.allocate_tensors()
in_d = interp.get_input_details()[0]; out_d = interp.get_output_details()[0]

y_true = []
y_pred = []
y_scores = []

for i in range(X.shape[0]):
    emb = X[i].astype("float32").reshape(1, -1)
    # quantization handling if needed
    if in_d["dtype"] != emb.dtype:
        # try naive mapping if quantized
        try:
            scale, zp = in_d.get("quantization", (0.0,0))
            if scale != 0:
                q = emb / scale + zp
                emb_in = q.round().astype(in_d["dtype"])
            else:
                emb_in = emb.astype(in_d["dtype"])
        except Exception:
            emb_in = emb.astype(in_d["dtype"])
    else:
        emb_in = emb

    interp.set_tensor(in_d["index"], emb_in)
    interp.invoke()
    out = interp.get_tensor(out_d["index"])
    probs = out.reshape(-1)
    idx = int(np.argmax(probs))
    reg_pred = inv_map.get(idx, str(idx))
    y_true.append(y[i].item())
    y_pred.append(reg_pred)
    y_scores.append(float(probs[idx]))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = float(accuracy_score(y_true, y_pred))
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

summary = {"accuracy": acc, "precision_macro": float(prec), "recall_macro": float(rec), "f1_macro": float(f1)}
(OUT / "classifier_eval.json").write_text(json.dumps(summary, indent=2))
# write confusion matrix CSV
import csv
labels = list(np.unique(y_true))
with open(OUT / "confusion_matrix.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([""] + labels)
    for i, row in enumerate(cm):
        w.writerow([labels[i]] + list(row.tolist()))

print("Wrote reports/classifier_eval.json and confusion_matrix.csv")
print("Summary:", summary)
