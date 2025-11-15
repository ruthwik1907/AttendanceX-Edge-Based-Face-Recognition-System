#!/usr/bin/env python3
# scripts/train_classifier_from_tflite_embeddings.py
"""
Compute embeddings from models/mobilefacenet.tflite for all images under data/processed/<REG>/
Save train_embeddings.npz and train a small Keras classifier which is exported to TFLite.
"""
import os, json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# tflite runtime
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

# keras/tf
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

MODELS = Path("models")
DATA = Path("data/processed")
OUT = MODELS
OUT.mkdir(parents=True, exist_ok=True)

EMB_TFLITE = MODELS / "mobilefacenet.tflite"
if not EMB_TFLITE.exists():
    raise SystemExit("mobilefacenet.tflite missing in models/")

# load tflite interpreter for embedding
emb_i = tflite.Interpreter(model_path=str(EMB_TFLITE))
emb_i.allocate_tensors()
emb_in = emb_i.get_input_details()[0]
emb_out = emb_i.get_output_details()[0]

# shape & dtype
_, H, W, C = emb_in["shape"]
emb_dtype = emb_in["dtype"]
emb_scale, emb_zero = emb_in.get("quantization", (0.0, 0))

print("Embedding model: input", emb_in["shape"], emb_dtype, "quant:", (emb_scale, emb_zero))
print("Embedding output shape:", emb_out["shape"], "dtype:", emb_out["dtype"])

# helper preprocessing (match earlier realtime script)
import cv2
def preprocess_for_tflite(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H)).astype("float32")
    float_input = img / 127.5 - 1.0
    # quantize if needed
    if emb_dtype == np.float32:
        return np.expand_dims(float_input.astype(np.float32), 0)
    else:
        # use quantization parameters if present
        if emb_scale == 0:
            est_scale = 1.0 / 127.0
            q = float_input / est_scale
            zp = 0
        else:
            q = float_input / emb_scale + emb_zero
            zp = emb_zero
        if np.issubdtype(np.dtype(emb_dtype), np.signedinteger):
            q = np.round(q).astype(np.int8)
        else:
            q = np.round(q).astype(np.uint8)
        return np.expand_dims(q, 0)

# collect images and labels
X_embs = []
Y = []
regs = sorted([d.name for d in DATA.iterdir() if d.is_dir()])
if not regs:
    raise SystemExit("No processed student folders under data/processed/")

for reg in regs:
    folder = DATA / reg
    imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    if not imgs:
        print("Skipping", reg, "no images")
        continue
    for p in imgs:
        try:
            img = cv2.imread(str(p))
            if img is None: 
                print("Warning: couldn't read", p); continue
            inp = preprocess_for_tflite(img)
            emb_i.set_tensor(emb_in["index"], inp)
            emb_i.invoke()
            emb_out_data = emb_i.get_tensor(emb_out["index"])
            # convert to float32 embeddings
            emb_f = emb_out_data.astype("float32")
            # flatten to 1D
            emb_flat = emb_f.reshape(-1)
            X_embs.append(emb_flat)
            Y.append(reg)
        except Exception as e:
            print("Error processing", p, e)

X = np.vstack(X_embs)
Y = np.array(Y)
print("Collected embeddings:", X.shape, "labels:", len(np.unique(Y)))

# save embeddings for record/training
np.savez(OUT / "train_embeddings.npz", X=X, y=Y)
print("Saved train_embeddings.npz")

# Train a small Keras classifier on embeddings
le = LabelEncoder(); y_int = le.fit_transform(Y)
num_classes = len(le.classes_)
y_oh = keras.utils.to_categorical(y_int, num_classes)

# simple MLP
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# train/test split
Xtr, Xv, ytr, yv = train_test_split(X, y_oh, test_size=0.12, random_state=42, stratify=y_int)
print("Training classifier on embeddings...")
model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=25, batch_size=32)

# save keras & convert to tflite (float)
keras_path = OUT / "classifier_keras.h5"
model.save(keras_path)
print("Saved keras model:", keras_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfl = converter.convert()
(OUT / "classifier.tflite").write_bytes(tfl)
print("Wrote models/classifier.tflite")

# label map
label_map = {str(i): lab for i, lab in enumerate(le.classes_)}
(OUT / "label_map.json").write_text(json.dumps(label_map, indent=2))
print("Wrote label_map.json")
