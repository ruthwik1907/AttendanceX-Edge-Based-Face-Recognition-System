# engine/phase3/detector_edgetpu.py
import cv2
import numpy as np

class EdgeTPUFaceDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # placeholder — actual TPU init later
        # DO NOT mix tracking / recognition here

    def detect(self, frame: np.ndarray):
        """
        Returns:
            List of bboxes: [(x1, y1, x2, y2, confidence)]
        """
        h, w, _ = frame.shape

        # 🔴 TEMP DUMMY OUTPUT (to validate pipeline)
        # Replace with real Edge TPU inference next
        return []
