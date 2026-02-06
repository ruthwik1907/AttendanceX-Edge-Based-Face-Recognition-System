# engine/phase3/detector_yolo.py
from ultralytics import YOLO
import cv2

class YOLOFaceDetector:
    def __init__(self, model_path, conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Returns:
        [(x1, y1, x2, y2, conf)]
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            verbose=False,
            device="cpu"   # Pi-safe
        )

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))

        return detections
