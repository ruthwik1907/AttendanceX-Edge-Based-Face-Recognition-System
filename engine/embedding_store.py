import numpy as np
from pathlib import Path
from typing import Union

class EmbeddingStore:
    def __init__(self, threshold=0.6):
        self.db = {}  # student_id -> (N,512)
        self.threshold = float(threshold)

    def load_from_disk(self, students_dir: Union[str, Path]):
        students_dir = Path(students_dir)

        for student_dir in students_dir.iterdir():
            emb_file = student_dir / "embeddings.npy"
            if emb_file.exists():
                embeddings = np.load(emb_file)
                self.db[student_dir.name] = embeddings

        print(f"[Store] Loaded {len(self.db)} students")

    def match(self, emb: np.ndarray):
        best_id = None
        best_score = 0.0

        for student_id, ref in self.db.items():
            sims = ref @ emb  # (N,)
            score = float(np.max(sims))

            if score > best_score:
                best_score = score
                best_id = student_id

        if best_score >= self.threshold:
            return best_id, best_score

        return None, best_score
