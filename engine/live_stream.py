# engine/live_stream.py

import cv2
import time
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import AttendanceSession

from engine.detector_yolo import YOLOFaceDetector
from engine.tracker import FaceTracker
from engine.recognizer_arcface import ArcFaceRecognizer
from engine.embedding_store import EmbeddingStore
from engine.track_identity import TrackIdentityManager
from engine.presence_tracker import PresenceTracker
from engine.attendance_evaluator import AttendanceEvaluator


def live_stream():
    db: Session = SessionLocal()

    detector = YOLOFaceDetector("models/yolov8n-face.pt", conf=0.4)
    tracker = FaceTracker()
    recognizer = ArcFaceRecognizer("models/arcface.onnx")

    store = EmbeddingStore(threshold=0.6)
    store.load_from_disk("data/students")

    identity_mgr = TrackIdentityManager(min_frames=3)
    presence_tracker = PresenceTracker()
    attendance_eval = AttendanceEvaluator(min_ratio=0.6)

    cap = None

    try:
        while True:
            session = db.query(AttendanceSession)\
                        .filter_by(is_active=True)\
                        .first()

            # ⛔ No active session → idle
            if not session:
                if cap:
                    cap.release()
                    cap = None
                time.sleep(0.5)
                continue

            # ✅ Start camera ONCE
            if cap is None:
                cap = cv2.VideoCapture(0)

            ret, frame = cap.read()
            if not ret:
                continue

            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            active_ids = set()

            for x1, y1, x2, y2, track_id in tracks:
                active_ids.add(track_id)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                identity = identity_mgr.get_identity(track_id)

                if identity is None:
                    emb = recognizer.get_embedding(face)
                    if emb is not None:
                        sid, score = store.match(emb)
                        if sid:
                            identity_mgr.update(track_id, sid, score)

                label = identity_mgr.get_identity(track_id)
                if label:
                    presence_tracker.update(label)

                color = (0, 255, 0) if label else (0, 200, 255)
                name = label if label else f"Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            identity_mgr.cleanup(active_ids)

            # 🎯 Encode frame for browser
            _, jpeg = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )

    finally:
        if cap:
            cap.release()
        db.close()
