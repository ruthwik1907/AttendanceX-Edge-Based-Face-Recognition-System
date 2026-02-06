# engine/phase3/utils.py
import time
import cv2
import numpy as np

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Input: BGR face crop (H, W, 3)
    Output: NHWC float32 (1, 112, 112, 3)
    """
    face = cv2.resize(face_bgr, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)  # (1, 112, 112, 3)
    return face

class FPSCounter:
    def __init__(self):
        self.prev = time.time()
        self.fps = 0.0

    def update(self):
        now = time.time()
        dt = now - self.prev
        self.prev = now
        if dt > 0:
            self.fps = 1.0 / dt
        return self.fps
