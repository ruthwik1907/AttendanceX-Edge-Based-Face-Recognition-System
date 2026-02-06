import cv2
import threading
import time
from pathlib import Path

from engine.detector_yolo import YOLOFaceDetector
from engine.tracker import FaceTracker
from engine.recognizer_arcface import ArcFaceRecognizer
from engine.embedding_store import EmbeddingStore
from engine.track_identity import TrackIdentityManager
from engine.presence_tracker import PresenceTracker
from engine.attendance_evaluator import AttendanceEvaluator



class PipelineRuntime:
    def __init__(self):
        self.running = False
        self.thread = None
        self.frame = None
        self.last_frame = None
        self.unknown_ids = set()

        self.detector = YOLOFaceDetector("models/yolov8n-face.pt", conf=0.4)
        self.tracker = FaceTracker()
        self.recognizer = ArcFaceRecognizer("models/arcface.onnx")

        self.store = EmbeddingStore(threshold=0.6)
        self.store.load_from_disk("data/students")

        self.identity_mgr = TrackIdentityManager()
        self.presence_tracker = PresenceTracker()
        self.attendance_eval = AttendanceEvaluator(min_ratio=0.6)

    def start(self):
        if self.running:
            return
        self.unknown_ids = set()
        self.last_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)

            for x1, y1, x2, y2, track_id in tracks:
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                identity = self.identity_mgr.get_identity(track_id)
                if identity is None:
                    emb = self.recognizer.get_embedding(face)
                    if emb is not None:
                        sid, score = self.store.match(emb)
                        if sid:
                            self.identity_mgr.update(track_id, sid, score)

                label = self.identity_mgr.get_identity(track_id)
                if label:
                    self.presence_tracker.update(label)
                else:
                    self.unknown_ids.add(track_id)

                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, label or "Unknown",
                            (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,0),2)

            self.frame = frame.copy()
            self.last_frame = frame.copy()

        cap.release()

    def get_frame(self):
        return self.frame

runtime = PipelineRuntime()
