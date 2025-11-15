# AttendanceX — Edge-Based Face Recognition System

**Last Updated:** November 16, 2025

A complete, production-ready face recognition attendance system built for institutional use. Designed for quick deployment on a laptop with web-based enrollment and real-time recognition.

## 📦 Package Contents

```
AttendanceX Project/
├── scripts/
│   ├── serve_uploads_polished.py      # Web uploader UI (students upload photos)
│   ├── process_all.py                 # Face detection & preprocessing
│   ├── train_embeddings_classifier.py # Train classifier on embeddings
│   ├── export_classifier_tflite.py    # Export to TFLite (edge inference)
│   ├── recognize_and_mark.py          # Live recognition (PyTorch, high-quality)
│   ├── realtime_tflite_infer.py       # Fast TFLite inference (mobile-ready)
│   ├── evaluate_model.py              # Accuracy & confusion matrix reports
│   └── automated_pipeline.py          # One-command full pipeline
├── models/
│   ├── mobilefacenet.tflite           # Pre-trained embedding extractor
│   ├── face_classifier.joblib         # (Generated) scikit-learn classifier
│   ├── classifier.tflite              # (Generated) TFLite classifier
│   └── label_map.json                 # (Generated) student ID mappings
├── data/
│   ├── enroll/                        # Raw student uploads (by Reg ID)
│   └── processed/                     # Preprocessed, augmented images
├── reports/                           # Quality reports & logs
├── requirements.txt                   # Python dependencies
├── studentMaster.xlsx                 # (Optional) student roster template
├── attendance.csv                     # Live attendance log
└── README.md                          # This file
```

## 🚀 Quick Start (5 minutes)

### 1. Set up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Troubleshooting:** If you hit `numpy` or `imgaug` conflicts, install `numpy<2.0.0`:
> ```bash
> pip install numpy==1.23.5
> pip install -r requirements.txt
> ```

### 3. Start the Web Uploader

```bash
python scripts/serve_uploads_polished.py --host 0.0.0.0 --port 5000
```

**Output:**
```
Starting AttendanceX server
Local:   http://127.0.0.1:5000
Network: http://<YOUR_PC_IP>:5000
```

Students visit `http://<YOUR_PC_IP>:5000` on their phones (same Wi‑Fi), enter their **9-digit Registration ID**, and upload **30–40 clear face photos**.

Photos are saved to: `data/enroll/<RegistrationID>/`

### 4. Process & Train

```bash
# Run full pipeline (preprocessing → training → export → evaluation)
python scripts/automated_pipeline.py --auto
```

**What it does:**
- ✅ Detects faces & extracts embeddings (using `mobilefacenet.tflite`)
- ✅ Augments images for robustness
- ✅ Trains a classifier on embeddings
- ✅ Exports to TFLite for edge inference
- ✅ Generates quality reports & metrics
- ✅ Creates `attendance.csv` template

**Outputs generated:**
- `models/face_classifier.joblib` — scikit-learn classifier
- `models/classifier.tflite` — TFLite classifier (fast inference)
- `models/label_map.json` — student ID mappings
- `models/embeddings_db.json` — student embeddings & thresholds
- `reports/evaluation_summary.json` — accuracy, precision, recall, F1
- `reports/confusion_matrix.csv` — per-student performance
- `reports/failed_detection_report.csv` — images where face detection failed
- `reports/upload_verification.json` — upload counts per student
- `attendance.csv` — live recognition log

### 5. Run Live Recognition

**High-quality (PyTorch + InceptionResnetV1):**
```bash
python scripts/recognize_and_mark.py
```
- Opens your laptop webcam
- Real-time face detection & recognition
- Logs to `attendance.csv`
- Best accuracy, slower inference

**Fast (TFLite on edge device):**
```bash
python scripts/realtime_tflite_infer.py
```
- Uses `mobilefacenet.tflite` + `models/classifier.tflite`
- Optimized for low-end devices & edge deployment
- Faster, lower memory footprint

---

## 📋 Demo Flow (for Viva / Presentation)

1. **Show enrollment:** Start uploader, upload 5 sample images on phone → show REC-themed UI
   ```bash
   python scripts/serve_uploads_polished.py --host 0.0.0.0 --port 5000
   ```

2. **Show preprocessing:** Run processing & open quality report
   ```bash
   python scripts/process_all.py
   # Open: reports/failed_detection_report.csv
   ```

3. **Show training:** Train classifier & export to TFLite
   ```bash
   python scripts/train_embeddings_classifier.py
   python scripts/export_classifier_tflite.py
   ```

4. **Show evaluation:** Display metrics & confusion matrix
   ```bash
   python scripts/evaluate_model.py
   # Open: reports/evaluation_summary.json, reports/confusion_matrix.csv
   ```

5. **Live demo:** Run recognition on webcam
   ```bash
   python scripts/recognize_and_mark.py
   # Show real-time detection & attendance logging
   ```

6. **Explain data quality:** Show `reports/upload_verification.json` & failed-detection report

---

## 🎨 UI Customization

The web uploader uses the **REC Color Scheme:**
- **Purple:** `#6A1B9A`
- **Gold:** `#FBC02D`
- **Cream:** `#FAF5EF`
- **Grey:** `#212121`

Edit `scripts/serve_uploads_polished.py` (HTML/CSS section) to customize.

---

## ⚙️ Configuration & Advanced Usage

### Adjust image requirements (in scripts)

**Registration ID validation:**
- Currently: **9 digits only** (e.g., `220101001`)
- Edit: `scripts/serve_uploads_polished.py` line ~350 (pattern)

**Photo upload folder:**
- Default: `data/enroll/<RegistrationID>/`
- Edit: `scripts/serve_uploads_polished.py` variable `UP`

### Adjust recognition thresholds

Edit `scripts/recognize_and_mark.py`:
```python
CONFIDENCE_THRESHOLD = 0.5  # Lower = more lenient, higher = stricter
```

### Run pipeline incrementally

The pipeline is **incremental**—it skips students already processed:
```bash
python scripts/process_all.py           # Process only new uploads
python scripts/train_embeddings_classifier.py  # Retrain on all
python scripts/export_classifier_tflite.py     # Export TFLite
python scripts/evaluate_model.py        # Evaluate performance
```

---

## 📊 Output Reports

After running the pipeline, check:

| File | Purpose |
|------|---------|
| `reports/evaluation_summary.json` | Accuracy, precision, recall, F1 per student |
| `reports/confusion_matrix.csv` | Misclassifications matrix |
| `reports/failed_detection_report.csv` | Images where MTCNN failed (ask for re-upload) |
| `reports/upload_verification.json` | Photo count per student |
| `reports/processed_counts.csv` | Preprocessed images per student |
| `attendance.csv` | Live recognition logs (timestamp, student, confidence) |

---

## 🛠️ Troubleshooting

### **Q: "No faces detected" in many images**
- **A:** Open `reports/failed_detection_report.csv` — ask students to re-upload with:
  - Better lighting
  - Direct front-facing angle
  - No extreme expressions or occlusion
  - Higher resolution

### **Q: Poor accuracy despite good images**
- **A:** Likely need more training images per student. Collect 40+ images with varied:
  - Angles (left, center, right profile)
  - Lighting conditions (indoor, outdoor, side-lit)
  - Expressions (neutral, slight smile)
  - Glasses on/off

### **Q: Recognition is slow**
- **A:** Use `realtime_tflite_infer.py` instead of `recognize_and_mark.py` (TFLite is ~10x faster)

### **Q: MTCNN GPU memory issues**
- **A:** Edit `scripts/process_all.py` and set `device='cpu'` in MTCNN initialization

### **Q: Attendance.csv not updating**
- **A:** Check file permissions; ensure `recognize_and_mark.py` has write access to project folder

---

## 💡 Tips & Best Practices

- **Collection:** Advise students to upload in **good lighting**, **clear angles**, **varied expressions**
- **Incremental:** After first run, re-uploads are automatically added to training
- **Privacy:** All data stored locally (no cloud uploads)
- **Backup:** Copy `models/` and `reports/` folders before re-running pipeline (to preserve history)
- **Edge deployment:** Use TFLite exports (`classifier.tflite`) for mobile/edge devices

---

## 📦 Dependencies

Core libraries:
- `torch`, `torchvision` — PyTorch (face embeddings)
- `facenet-pytorch` — MTCNN + InceptionResnetV1
- `tensorflow`, `tflite-runtime` — TFLite inference
- `scikit-learn` — Classifier training
- `opencv-python` — Image processing
- `pandas`, `openpyxl` — Data handling
- `flask` — Web uploader UI

See `requirements.txt` for full list with versions.

---

## 📝 License & Credits

**AttendanceX** — REC institutional attendance system.

Built with:
- MTCNN (face detection)
- InceptionResnetV1 (face embeddings)
- MobileFaceNet TFLite (edge embeddings)
- scikit-learn (classification)

---

## ❓ Need Help?

- **Stuck on setup?** Check Python version (3.8+), verify virtual env is activated
- **Models missing?** Run `automated_pipeline.py --auto` to generate
- **Want to add features?** Open an issue or contact the maintainer

---

**Last Updated:** November 16, 2025  
**Status:** ✅ Ready for production / viva demo  
**Good luck!** 🎓
