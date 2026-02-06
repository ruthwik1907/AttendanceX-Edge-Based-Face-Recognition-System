# engine/registration.py

from pathlib import Path
import cv2
import numpy as np

from engine.detector_yolo import YOLOFaceDetector
from engine.recognizer_arcface import ArcFaceRecognizer


class RegistrationEngine:
    def __init__(
        self,
        detector_model: str,
        recognizer_model: str,
        min_faces: int = 10
    ):
        """
        detector_model: path to YOLO face model
        recognizer_model: path to ArcFace ONNX model
        min_faces: minimum valid face crops required
        """
        self.detector = YOLOFaceDetector(detector_model)
        self.recognizer = ArcFaceRecognizer(recognizer_model)
        self.min_faces = min_faces

    def register_student(
        self,
        student_id: str,
        name: str,
        images: list[Path]
    ):
        """
        images: list of image Paths
        returns: dict with samples_used + embedding path
        """

        student_dir = Path("data/students") / student_id
        student_dir.mkdir(parents=True, exist_ok=True)

        all_embeddings = []

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            detections = self.detector.detect(img)
            if not detections:
                continue

            # take largest face
            det = max(
                detections,
                key=lambda d: (d[2] - d[0]) * (d[3] - d[1])
            )

            x1, y1, x2, y2 = map(int, det[:4])
            
            # Safety clipping
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue

            emb = self.recognizer.get_embedding(face)
            if emb is None:
                continue

            all_embeddings.append(emb)

        if len(all_embeddings) < self.min_faces:
            raise ValueError("Not enough valid face embeddings")

        embeddings = np.vstack(all_embeddings).astype("float32")

        # L2 normalize (CRITICAL)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        np.save(student_dir / "embeddings.npy", embeddings)

        return {
            "student_id": student_id,
            "name": name,
            "samples_used": len(all_embeddings),
            "embedding_path": str(student_dir / "embeddings.npy")
        }
