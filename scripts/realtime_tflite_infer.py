#!/usr/bin/env python3
# scripts/realtime_tflite_infer_safe2.py
"""
Robust realtime ArcFace ONNX inference helper.
- Uses ArcFace ONNX model for embeddings and joblib classifier.
- Safely handles weird tensor shapes returned by interpreters.
- Prints a single diagnostic when it sees an unexpected shape/type.
- Falls back to headless (no GUI) if cv2.imshow isn't available.
"""
import cv2, numpy as np, json, time, sys, traceback
from pathlib import Path
import onnxruntime as ort
import joblib

MODELS = Path("models")
EMB_PATH = MODELS / "arcface.onnx"
CLS_PATH = MODELS / "classifier_arcface.joblib"
LABEL_MAP_PATH = MODELS / "label_map_arcface.json"

if not EMB_PATH.exists() or not CLS_PATH.exists() or not LABEL_MAP_PATH.exists():
    print("Missing required files in models/: arcface.onnx, classifier_arcface.joblib, label_map_arcface.json")
    sys.exit(1)

label_map = json.loads(LABEL_MAP_PATH.read_text())

# load ONNX session for ArcFace embeddings
emb_session = ort.InferenceSession(str(EMB_PATH), providers=['CPUExecutionProvider'])
emb_input_name = emb_session.get_inputs()[0].name
emb_output_name = emb_session.get_outputs()[0].name
emb_input_shape = emb_session.get_inputs()[0].shape

# load joblib classifier
classifier = joblib.load(str(CLS_PATH))

print("ArcFace input shape:", emb_input_shape)
print("ArcFace input name:", emb_input_name)
print("Classifier loaded:", type(classifier))

# basic params for ArcFace (typically NCHW: batch, channels, height, width)
# Input shape is ['None', 3, 112, 112] = (batch, channels, height, width)
if len(emb_input_shape) == 4:
    # NCHW format: batch, channels, height, width
    _, C, H, W = emb_input_shape
    # Handle 'None' or dynamic batch dimension
    if isinstance(H, str): H = 112
    if isinstance(W, str): W = 112
    if isinstance(C, str): C = 3
else:
    H, W, C = 112, 112, 3  # default ArcFace size

print(f"Using H={H}, W={W}, C={C}")

def preprocess_frame(frame_bgr):
    # ArcFace preprocessing: RGB, resize to 112x112, normalize to [-1, 1]
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H)).astype(np.float32)
    # ArcFace normalization: (pixel / 127.5) - 1.0
    img = (img / 127.5) - 1.0
    # Transpose from HWC to CHW format for ONNX (batch, channels, height, width)
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0).astype(np.float32)

def safe_to_numpy(x):
    # convert whatever tflite returns (lists/tuples/scalars) to numpy array
    try:
        return np.asarray(x)
    except Exception:
        return np.array([x])

# helper to check whether cv2.imshow is usable
def can_show():
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        cv2.waitKey(1)  # Process window events
        return True
    except Exception:
        return False

show_gui = can_show()
print(f"GUI mode: {'enabled' if show_gui else 'disabled (headless)'}")

# open camera: on Windows prefer CAP_DSHOW
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    sys.exit(1)

# Create window early if GUI is enabled
if show_gui:
    cv2.namedWindow("ArcFace Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ArcFace Live", 800, 600)
    print("Window 'ArcFace Live' created.")

last_seen = {}
ATTENDANCE_LOG = Path("attendance.csv")
CONF_THRESHOLD = 0.60

# We'll print unusual-shape diagnostics only once to avoid spamming
diagnostic_shown = False

print("Starting ArcFace inference loop — press 'q' in the window (if shown) to quit.")
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
            # Run ArcFace ONNX inference
            emb_arr = emb_session.run([emb_output_name], {emb_input_name: inp})[0]
            emb_arr = emb_arr.astype(np.float32).reshape(1, -1)

            # Run classifier (joblib model - likely sklearn)
            # Most sklearn classifiers have predict_proba method
            if hasattr(classifier, 'predict_proba'):
                probs_raw = classifier.predict_proba(emb_arr)[0]
            else:
                # fallback: use predict and create one-hot style output
                pred = classifier.predict(emb_arr)[0]
                num_classes = len(label_map)
                probs_raw = np.zeros(num_classes)
                if pred < num_classes:
                    probs_raw[pred] = 1.0
            
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
                cv2.imshow("ArcFace Live", frame)
                # Must call waitKey to update window
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # headless
                print(txt)
                time.sleep(0.03)  # Small delay to prevent CPU overload

            # log attendance
            now = time.time()
            if conf >= CONF_THRESHOLD and (reg not in last_seen or now - last_seen[reg] > 30):
                last_seen[reg] = now
                ATTENDANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(ATTENDANCE_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{reg},{conf:.3f},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

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
