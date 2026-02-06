from typing import Optional
import numpy as np
import cv2
import onnxruntime as ort


class ArcFaceRecognizer:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("[Recognizer] ONNX input shape:",
              self.session.get_inputs()[0].shape)

    def get_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            face = cv2.resize(face_bgr, (112, 112))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype(np.float32) / 255.0
            face = np.expand_dims(face, axis=0)  # (1,112,112,3)

            emb = self.session.run(
                [self.output_name],
                {self.input_name: face}
            )[0][0]

            return emb / np.linalg.norm(emb)

        except Exception as e:
            print("[Recognizer] Failed:", e)
            return None
