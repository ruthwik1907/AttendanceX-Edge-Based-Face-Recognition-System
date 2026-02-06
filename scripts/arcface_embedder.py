# scripts/arcface_embedder.py
import onnxruntime as ort
import numpy as np
import cv2

class ArcFaceEmbedder:
    """
    ArcFace (ResNet100) embedding class using ONNXRuntime.
    Model input: [1, 3, 112, 112]
    Preprocessing: BGR → RGB, resize → 112×112, normalize to [-1,1]
    Output: 512-d float embedding
    """

    def __init__(self, model_path="models/arcface_r100.onnx"):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img = img.astype(np.float32) / 127.5 - 1.0  # [-1,1]
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(img, axis=0)

    def embed(self, img_bgr):
        inp = self.preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: inp})[0]
        emb = out[0]
        # L2 normalize output (ArcFace standard)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)
