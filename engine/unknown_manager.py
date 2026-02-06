import numpy as np


class UnknownManager:
    def __init__(self):
        self.store = {}

    def add(self, track_id, embedding):
        self.store.setdefault(track_id, []).append(embedding)

    def promote(self, track_id, student_id, db):
        avg = np.mean(self.store[track_id], axis=0)
        db.add_embedding(student_id, avg)
        del self.store[track_id]
