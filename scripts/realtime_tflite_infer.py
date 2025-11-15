#!/usr/bin/env python3
# scripts/realtime_tflite_infer_safe2.py
"""
Robust realtime TFLite inference helper.
- Safely handles weird tensor shapes returned by interpreters.
- Prints a single diagnostic when it sees an unexpected shape/type.
- Falls back to headless (no GUI) if cv2.imshow isn't available.
"""
import cv2, numpy as np, json, time, sys, traceback
from pathlib import Path

# Prefer tflite_runtime if available
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

MODELS = Path("models")
EMB_PATH = MODELS / "mobilefacenet.tflite"
CLS_PATH = MODELS / "classifier.tflite"
LABEL_MAP_PATH = MODELS / "label_map.json"

if not EMB_PATH.exists() or not CLS_PATH.exists() or not LABEL_MAP_PATH.exists():
    print("Missing required files in models/: mobilefacenet.tflite, classifier.tflite, label_map.json")
    sys.exit(1)

label_map = json.loads(LABEL_MAP_PATH.read_text())

# load interpreters
emb_i = tflite.Interpreter(model_path=str(EMB_PATH)); emb_i.allocate_tensors()
emb_in = emb_i.get_input_details()[0]; emb_out = emb_i.get_output_details()[0]

cls_i = tflite.Interpreter(model_path=str(CLS_PATH)); cls_i.allocate_tensors()
cls_in = cls_i.get_input_details()[0]; cls_out = cls_i.get_output_details()[0]

print("Embedding input:", emb_in["shape"], emb_in["dtype"], "quant:", emb_in.get("quantization", None))
print("Embedding output:", emb_out["shape"], emb_out["dtype"])
print("Classifier input:", cls_in["shape"], cls_in["dtype"], "quant:", cls_in.get("quantization", None))
print("Classifier output:", cls_out["shape"], cls_out["dtype"])

# basic params
_, H, W, C = emb_in["shape"]
emb_dtype = np.dtype(emb_in["dtype"])
emb_quant = emb_in.get("quantization", (0.0, 0))
cls_dtype = np.dtype(cls_in["dtype"])
cls_quant = cls_in.get("quantization", (0.0, 0))

def preprocess_frame(frame_bgr):
    # convert center crop if no face detector
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H)).astype(np.float32)
    float_input = img / 512 - 1.0  # common for mobilefacenet
    if emb_dtype == np.float32:
        return np.expand_dims(float_input.astype(np.float32), 0)
    # quantized input expected
    scale, zp = emb_quant
    if scale == 0:
        # fallback heuristic: map [-1,1] to int8 range
        est_scale = 1.0 / 512.0
        q = float_input / est_scale
        zp = 0
    else:
        q = float_input / scale + zp
    if np.issubdtype(emb_dtype, np.signedinteger):
        q = np.round(q).astype(np.int8)
    else:
        q = np.round(q).astype(np.uint8)
    return np.expand_dims(q, 0)

def safe_to_numpy(x):
    # convert whatever tflite returns (lists/tuples/scalars) to numpy array
    try:
        return np.asarray(x)
    except Exception:
        return np.array([x])

# helper to check whether cv2.imshow is usable
def can_show():
    try:
        cv2.namedWindow("test")
        cv2.destroyWindow("test")
        return True
    except Exception:
        return False

show_gui = can_show()
# open camera: on Windows prefer CAP_DSHOW
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    sys.exit(1)

last_seen = {}
ATTENDANCE_LOG = Path("attendance.csv")
CONF_THRESHOLD = 0.60

# We'll print unusual-shape diagnostics only once to avoid spamming
diagnostic_shown = False

print("Starting inference loop — press 'q' in the window (if shown) to quit.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed — exiting.")
            break

        # preprocess (center-crop fallback)
        h0,w0 = frame.shape[:2]
        side = min(h0,w0)
        cy, cx = h0//2, w0//2
        crop = frame[cy-side//2:cy+side//2, cx-side//2:cx+side//2]

        inp = preprocess_frame(crop)

        try:
            emb_i.set_tensor(emb_in["index"], inp)
            emb_i.invoke()
            emb_raw = emb_i.get_tensor(emb_out["index"])
            emb_arr = safe_to_numpy(emb_raw).astype(np.float32).reshape(1, -1)

            # double-check dims vs classifier input
            expected_cls_dim = int(np.prod(cls_in["shape"][1:])) if len(cls_in["shape"])>1 else cls_in["shape"][0]
            if emb_arr.shape[1] != expected_cls_dim:
                # show a single diagnostic and stop — mismatch should be handled by training script
                if not diagnostic_shown:
                    print("DIAGNOSTIC: embedding dim", emb_arr.shape, "does not match classifier input dim", cls_in["shape"])
                    print("Solution: train a classifier using these embeddings (scripts/train_classifier_from_tflite_embeddings.py).")
                    diagnostic_shown = True
                # to avoid repeated noisy errors, pause briefly and continue
                time.sleep(0.5)
                continue

            # prepare classifier input: classifier may expect float32 or quantized
            if cls_dtype == np.float32:
                cls_input = emb_arr.astype(np.float32)
            else:
                # quantize embedding to class input dtype
                scale, zp = cls_quant
                if scale == 0:
                    est_scale = 1.0 / 512.0
                    q = emb_arr / est_scale
                    zp = 0
                else:
                    q = emb_arr / scale + zp
                if np.issubdtype(cls_dtype, np.signedinteger):
                    q = np.round(q).astype(np.int8)
                else:
                    q = np.round(q).astype(np.uint8)
                cls_input = q

            cls_i.set_tensor(cls_in["index"], cls_input)
            cls_i.invoke()
            probs_raw = cls_i.get_tensor(cls_out["index"])
            probs = safe_to_numpy(probs_raw)

            # normalize probs shape: could be (1,N) or (N,) or scalar
            if probs.ndim == 2 and probs.shape[0] == 1:
                probs = probs[0]
            elif probs.ndim == 0:
                # single value — wrap into array
                probs = np.array([probs.item()])
            elif probs.ndim > 2:
                # unexpected higher dims — show a diagnostic once and reshape if possible
                if not diagnostic_shown:
                    print("DIAGNOSTIC: classifier returned unexpected tensor shape", probs.shape)
                    print("Attempting to flatten the probabilities to 1-D.")
                    diagnostic_shown = True
                probs = probs.reshape(-1)

            # finally safe argmax
            if probs.size == 0:
                print("Warning: classifier returned empty probabilities — skipping this frame.")
                time.sleep(0.05)
                continue

            try:
                idx = int(np.argmax(probs))
            except Exception as e:
                # very defensive: print small diagnostic, skip frame
                if not diagnostic_shown:
                    print("DIAGNOSTIC: np.argmax failed on probs with dtype", probs.dtype, "and shape", probs.shape)
                    print("Exception:", e)
                    diagnostic_shown = True
                time.sleep(0.05)
                continue

            conf = float(probs[idx]) if idx < probs.shape[0] else float(probs.max())
            reg = label_map.get(str(idx), str(idx))

            # annotate/display or headless printing
            txt = f"{reg} ({conf:.2f})"
            if show_gui:
                cv2.putText(frame, txt, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0) if conf>=CONF_THRESHOLD else (0,120,255), 2)
                cv2.imshow("TFLite Live", frame)
            else:
                # headless
                print(txt)

            # log attendance
            now = time.time()
            if conf >= CONF_THRESHOLD and (reg not in last_seen or now - last_seen[reg] > 30):
                last_seen[reg] = now
                ATTENDANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(ATTENDANCE_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{reg},{conf:.3f},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # handle GUI quit
            if show_gui and (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        except Exception as e:
            # Print a short traceback only first time, then a short message to avoid flooding console
            if not diagnostic_shown:
                print("Inference exception (first occurrence):")
                traceback.print_exc()
                diagnostic_shown = True
            else:
                # short message
                print("Inference exception (repeated) — skipping frame.")
            time.sleep(0.08)
            continue

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    if show_gui:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    print("Exiting cleanly.")
