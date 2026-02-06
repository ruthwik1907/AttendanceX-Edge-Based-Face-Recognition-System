# engine/phase3/tracker.py
import numpy as np
from norfair import Detection, Tracker

class FaceTracker:
    def __init__(self):
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=50
        )

    def update(self, detections):
        norfair_dets = []

        for x1, y1, x2, y2, conf in detections:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            norfair_dets.append(
                Detection(points=np.array([[cx, cy]]))
            )

        tracked = self.tracker.update(norfair_dets)

        results = []
        for obj in tracked:
            if obj.id is None:
                continue

            cx, cy = obj.estimate[0]
            size = 120
            results.append((
                int(cx - size),
                int(cy - size),
                int(cx + size),
                int(cy + size),
                obj.id
            ))

        return results
