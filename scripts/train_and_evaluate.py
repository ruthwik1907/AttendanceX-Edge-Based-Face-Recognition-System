#!/usr/bin/env python3
"""
train_and_evaluate.py — ArcFace version (fixed argparse attribute names)

Usage:
    python scripts/train_and_evaluate.py --data-dir data/processed --models-dir models --normalize --test-size 0.25
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import joblib

# local embedder (ArcFace ONNX)
from arcface_embedder import ArcFaceEmbedder

# ------------------------------------------------------------
# Util
# ------------------------------------------------------------
# ---------------- Add these embedder wrapper classes ----------------
# PyTorch FaceNet wrapper (uses facenet-pytorch if installed)
class PyTorchEmbeddingModel:
    def __init__(self, device=None):
        try:
            from facenet_pytorch import InceptionResnetV1
            import torch
        except Exception as e:
            raise RuntimeError("facenet-pytorch not installed. Install with: pip install facenet-pytorch torch torchvision") from e
        self.device = device or ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def preprocess(self, img_bgr):
        from PIL import Image
        import torchvision.transforms as transforms
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb).convert('RGB').resize((160,160))
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
        return tfm(pil).unsqueeze(0).to(self.device)

    def embed(self, img_bgr):
        import torch
        x = self.preprocess(img_bgr)
        with torch.no_grad():
            emb = self.model(x).cpu().numpy()
        return emb.reshape(-1).astype(np.float32)

# TFLite wrapper (quantized or float TFLite)
class TFLiteEmbeddingModel:
    def __init__(self, model_path):
        try:
            import tensorflow as tf
            self.interp = tf.lite.Interpreter(model_path=str(model_path))
        except Exception:
            # try tflite_runtime
            import tflite_runtime.interpreter as tflite_rt
            self.interp = tflite_rt.Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]
        self.in_shape = tuple(self.inp['shape'])
        self.in_dtype = np.dtype(self.inp['dtype'])
        self.out_dtype = np.dtype(self.outp['dtype'])
        # infer H,W from shape e.g. [1,112,112,3] or [1,3,112,112]
        if len(self.in_shape) == 4 and self.in_shape[1] in (112,160):
            if self.in_shape[1] == 3:
                # channels-first
                self.H = self.in_shape[2]; self.W = self.in_shape[3]
            else:
                self.H = self.in_shape[1]; self.W = self.in_shape[2]
        else:
            # default
            self.H = 112; self.W = 112

    def preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.W, self.H)).astype(np.float32)
        arr = img / 127.5 - 1.0
        # if model expects NHWC
        if self.inp['shape'][-1] == 3:
            return np.expand_dims(arr.astype(np.float32), 0)
        # else NCHW
        return np.expand_dims(np.transpose(arr, (2,0,1)).astype(np.float32), 0)

    def embed(self, img_bgr):
        inp = self.preprocess(img_bgr)
        self.interp.set_tensor(self.inp['index'], inp)
        self.interp.invoke()
        out = self.interp.get_tensor(self.outp['index'])
        out = out.astype(np.float32)
        # flatten
        return out.reshape(-1)

# ONNX wrapper for MobileFaceNet-like models
class MobileFaceNetONNX:
    def __init__(self, model_path):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError("onnxruntime is required for ONNX models. Install with: pip install onnxruntime") from e
        self.sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        inp_shape = self.sess.get_inputs()[0].shape
        # determine H,W from shape (assume [1,3,H,W] or [1,H,W,3])
        if len(inp_shape) == 4:
            if inp_shape[1] == 3:
                self.H = inp_shape[2]; self.W = inp_shape[3]
                self.nchw = True
            else:
                self.H = inp_shape[1]; self.W = inp_shape[2]
                self.nchw = False
        else:
            self.H = 112; self.W = 112
            self.nchw = False

    def preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.W, self.H)).astype(np.float32)
        img = img / 127.5 - 1.0
        if self.nchw:
            img = np.transpose(img, (2,0,1))
        return np.expand_dims(img.astype(np.float32), 0)

    def embed(self, img_bgr):
        inp = self.preprocess(img_bgr)
        out = self.sess.run(None, {self.input_name: inp})[0]
        return out.reshape(-1).astype(np.float32)
# ---------------- end wrappers ----------------

def list_image_files(folder: Path):
    exts = ('.jpg','.jpeg','.png','.bmp')
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]

def l2normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n==0] = 1
    return x / n

def compute_eer(fpr, tpr, thr):
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thr[idx]

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # p = argparse.ArgumentParser()
    # p.add_argument('--data-dir', default='data/processed', help='per-person folders')
    # p.add_argument('--models-dir', default='models', help='models directory (arcface_r100.onnx expected here)')
    # p.add_argument('--test-size', type=float, default=0.25, help='per-person test fraction')
    # p.add_argument('--normalize', action='store_true', help='L2 normalize embeddings (recommended for ArcFace)')
    # p.add_argument('--pca-dim', type=int, default=0, help='PCA output dim (0 = disabled)')
    # p.add_argument('--calibrate', action='store_true', help='Calibrate similarity scores using logistic regression')
    # p.add_argument('--random-seed', type=int, default=42)
    # args = p.parse_args()

    # # NOTE: argparse converts flags with hyphens to args.attribute_name where hyphens -> underscores
    # DATA = Path(args.data_dir)
    # MODELS = Path(args.models_dir)
    # REPORTS = Path("reports")
    # MODELS.mkdir(parents=True, exist_ok=True)
    # REPORTS.mkdir(parents=True, exist_ok=True)

    # # verify arcface model exists
    # arcface_path = MODELS / "arcface.onnx"
    # if not arcface_path.exists():
    #     print(f"ERROR: ArcFace ONNX model not found at {arcface_path}")
    #     print("Place arcface_r100.onnx into the models/ directory and try again.")
    #     sys.exit(1)

    # embedder = ArcFaceEmbedder(str(arcface_path))

    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/processed', help='per-person folders')
    p.add_argument('--models-dir', default='models', help='models directory (arcface_r100.onnx expected here)')
    p.add_argument('--test-size', type=float, default=0.25, help='per-person test fraction')
    p.add_argument('--normalize', action='store_true', help='L2 normalize embeddings (recommended for ArcFace)')
    p.add_argument('--pca-dim', type=int, default=0, help='PCA output dim (0 = disabled)')
    p.add_argument('--calibrate', action='store_true', help='Calibrate similarity scores using logistic regression')
    p.add_argument('--embedder', choices=['arcface_onnx','facenet_pytorch','mobilefacenet_onnx','mobilefacenet_tflite'], default='arcface_onnx',
                help='Which embedding backend to use')
    p.add_argument('--random-seed', type=int, default=42)
    args = p.parse_args()

    DATA = Path(args.data_dir)
    MODELS = Path(args.models_dir)
    REPORTS = Path("reports")
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Choose embedding model based on --embedder
    emb_model = None
    if args.embedder == 'arcface_onnx':
        # accept either arcface_r100.onnx or arcface.onnx (backwards compat)
        cand1 = MODELS / "arcface_r100.onnx"
        cand2 = MODELS / "arcface.onnx"
        if cand1.exists():
            arcpath = cand1
        elif cand2.exists():
            arcpath = cand2
        else:
            print(f"ERROR: ArcFace ONNX model not found at {cand1} or {cand2}")
            print("Place arcface_r100.onnx into the models/ directory and try again.")
            sys.exit(1)
        from arcface_embedder import ArcFaceEmbedder
        emb_model = ArcFaceEmbedder(str(arcpath))
        print("Using ArcFace ONNX:", arcpath)

    elif args.embedder == 'facenet_pytorch':
        try:
            from facenet_pytorch import InceptionResnetV1
            import torch
        except Exception as e:
            print("ERROR: facenet-pytorch is not installed. Install with: pip install facenet-pytorch torch torchvision")
            sys.exit(1)
        # fallback to using the PyTorch wrapper implemented earlier in your repo (PyTorchEmbeddingModel)
        from train_and_evaluate import PyTorchEmbeddingModel  # assumes class exists in file; else import from helper
        emb_model = PyTorchEmbeddingModel()
        print("Using facenet-pytorch InceptionResnetV1")

    elif args.embedder == 'mobilefacenet_onnx':
        # expects models/mobilefacenet.onnx or models/mobilefacenet_fp.onnx
        m1 = MODELS / "mobilefacenet.onnx"
        m2 = MODELS / "mobilefacenet_fp.onnx"
        if m1.exists():
            mfpath = m1
        elif m2.exists():
            mfpath = m2
        else:
            print("ERROR: mobilefacenet ONNX not found at models/mobilefacenet.onnx (or mobilefacenet_fp.onnx).")
            sys.exit(1)
        # A small ONNX wrapper like ArcFaceEmbedder should be implemented; assume you have `MobileFaceNetONNX` class
        from mobilefacenet_embedder import MobileFaceNetONNX  # implement similar to arcface_embedder
        emb_model = MobileFaceNetONNX(str(mfpath))
        print("Using MobileFaceNet ONNX:", mfpath)

    elif args.embedder == 'mobilefacenet_tflite':
        # expects models/mobilefacenet.tflite
        tfpath = MODELS / "mobilefacenet.tflite"
        if not tfpath.exists():
            print("ERROR: mobilefacenet.tflite not found in models/. Place tflite model and retry.")
            sys.exit(1)
        # assume you have TFLiteEmbeddingModel implemented earlier
        from train_and_evaluate import TFLiteEmbeddingModel  # or proper import location
        emb_model = TFLiteEmbeddingModel(str(tfpath))
        print("Using MobileFaceNet TFLite:", tfpath)

    # collect persons
    persons = sorted([d.name for d in DATA.iterdir() if d.is_dir()])
    per_person = {}
    for p_name in persons:
        imgs = list_image_files(DATA / p_name)
        if len(imgs) >= 3:   # allow smaller sets but you can tune
            per_person[p_name] = imgs

    if not per_person:
        print("No person folders with images found under", DATA)
        sys.exit(1)

    rng = np.random.RandomState(args.random_seed)

    # ------------------------------------------------------------
    # per-person split
    # ------------------------------------------------------------
    train_items = []
    test_items = []
    for reg, imgs in per_person.items():
        imgs = np.array(imgs)
        n = len(imgs)
        k_test = max(1, int(math.ceil(n * args.test_size)))
        idx = np.arange(n)
        rng.shuffle(idx)
        test_idx = idx[:k_test]
        train_idx = idx[k_test:]
        # ensure at least one train sample
        if len(train_idx) == 0:
            train_idx = test_idx[:1]
            test_idx = test_idx[1:]
            if len(test_idx) == 0:
                # can't split
                continue
        for i in train_idx:
            train_items.append((str(imgs[i]), reg))
        for i in test_idx:
            test_items.append((str(imgs[i]), reg))

    print("Train items:", len(train_items), " Test items:", len(test_items))

    # ------------------------------------------------------------
    # compute embeddings helper
    # ------------------------------------------------------------
    def embed_set(items, desc):
        X = []
        Y = []
        for path, lab in tqdm(items, desc=desc):
            img = cv2.imread(path)
            if img is None:
                print("Warning: could not read", path)
                continue
            # use the selected embedding model instance called `emb_model`
            try:
                emb = emb_model.embed(img)
            except Exception as e:
                print(f"ERROR: embedding failed for {path}: {e}")
                continue
            X.append(emb)
            Y.append(lab)
        if len(X) == 0:
            return np.zeros((0,0)), np.array([])
        return np.vstack(X), np.array(Y)

    Xtr, ytr = embed_set(train_items, "embed-train")
    Xte, yte = embed_set(test_items, "embed-test")

    if Xtr.size == 0 or Xte.size == 0:
        print("Error: empty embeddings (check image files / paths). Aborting.")
        sys.exit(1)

    # L2 normalize (ArcFace typical)
    if args.normalize:
        Xtr = l2normalize(Xtr)
        Xte = l2normalize(Xte)

    # PCA optional
    pca = None
    if args.pca_dim and 0 < args.pca_dim < Xtr.shape[1]:
        print("Applying PCA ->", args.pca_dim)
        pca = PCA(n_components=args.pca_dim, whiten=True, random_state=args.random_seed)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)

    # Save embeddings for reproducibility
    np.savez(MODELS / "train_eval_arcface_embeddings.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print("Saved embeddings to models/train_eval_arcface_embeddings.npz")

    # ------------------------------------------------------------
    # Train classifier (identification)
    # ------------------------------------------------------------
    le = LabelEncoder()
    ytr_i = le.fit_transform(ytr)
    yte_i = le.transform(yte)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(Xtr, ytr_i)

    joblib.dump(clf, MODELS / "classifier_arcface.joblib")
    (MODELS / "label_map_arcface.json").write_text(json.dumps({str(i): c for i, c in enumerate(le.classes_)}))
    print("Saved classifier and label map to models/")

    # classification evaluation
    ypred_i = clf.predict(Xte)
    acc = accuracy_score(yte_i, ypred_i)
    cm = confusion_matrix(yte_i, ypred_i)
    creport = classification_report(yte_i, ypred_i, target_names=le.classes_, output_dict=True)
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(REPORTS / "cm_arcface.csv")
    (REPORTS / "classification_arcface.json").write_text(json.dumps(creport, indent=2))
    print("Classification accuracy:", acc)

    # ------------------------------------------------------------
    # Verification (pairwise cosine)
    # ------------------------------------------------------------
    def cosine_sim(a, b):
        # a: (D,) b:(D,) -> scalar
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    scores = []
    labels = []
    N = Xte.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            s = cosine_sim(Xte[i], Xte[j])
            scores.append(s)
            labels.append(1 if yte[i] == yte[j] else 0)
    scores = np.array(scores)
    labels = np.array(labels)

    # calibration if requested (train logistic on train pairs)
    if args.calibrate:
        print("Training logistic calibration on training pairs...")
        train_scores = []
        train_labels = []
        M = Xtr.shape[0]
        for i in range(M):
            for j in range(i + 1, M):
                s = cosine_sim(Xtr[i], Xtr[j])
                train_scores.append([s])
                train_labels.append(1 if ytr[i] == ytr[j] else 0)
        train_scores = np.array(train_scores)
        train_labels = np.array(train_labels)
        if len(np.unique(train_labels)) > 1:
            lr = LogisticRegression(max_iter=1000)
            # balance negatives/positives by sampling if needed
            lr.fit(train_scores, train_labels)
            scores = lr.predict_proba(scores.reshape(-1, 1))[:, 1]
            joblib.dump(lr, MODELS / "calibration_arcface.joblib")
            print("Saved calibration model.")
        else:
            print("Calibration skipped: not enough positive/negative pair variety in training set.")

    # ROC/AUC/EER
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_thr = compute_eer(fpr, tpr, thr)

    pd.DataFrame({"fpr": fpr, "tpr": tpr, "thr": thr}).to_csv(REPORTS / "roc_arcface.csv", index=False)
    (REPORTS / "verification_arcface.json").write_text(json.dumps({
        "auc": float(roc_auc),
        "eer": float(eer),
        "eer_threshold": float(eer_thr),
        "n_pairs": int(len(scores))
    }, indent=2))

    print(f"Verification AUC: {roc_auc:.4f}  EER: {eer:.4f} (thr {eer_thr:.4f})")

    # summary
    summary = {
        "accuracy": float(acc),
        "auc": float(roc_auc),
        "eer": float(eer),
        "classes": int(len(le.classes_)),
        "train": int(Xtr.shape[0]),
        "test": int(Xte.shape[0])
    }
    (REPORTS / "summary_arcface.json").write_text(json.dumps(summary, indent=2))
    print("Saved summary.")

if __name__ == "__main__":
    main()
