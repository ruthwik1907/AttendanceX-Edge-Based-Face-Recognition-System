import time

class Metrics:
    frames = 0
    start = time.time()
    detections = 0
    recognitions = 0

    @classmethod
    def fps(cls):
        return cls.frames / max(1, time.time() - cls.start)
