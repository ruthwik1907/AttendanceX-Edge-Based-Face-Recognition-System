from pathlib import Path
import numpy as np

BASE = Path("data/students")

for student in BASE.iterdir():
    emb_file = student / "embeddings.npy"
    if not emb_file.exists():
        print(f"[❌] {student.name}: NO embeddings.npy")
        continue

    emb = np.load(emb_file)

    if emb.ndim == 1:
        print(f"[❌] {student.name}: INVALID shape {emb.shape}")
        continue

    norms = np.linalg.norm(emb, axis=1)
    print(
        f"[OK] {student.name}: "
        f"shape={emb.shape}, "
        f"norm(mean)={norms.mean():.3f}"
    )
