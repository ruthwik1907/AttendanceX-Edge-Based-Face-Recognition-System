import time
from collections import defaultdict


class AttendanceManager:
    def __init__(self, min_duration=60):
        self.min_duration = min_duration
        self.active_tracks = {}
        self.confirmed = set()

    def update(self, student_id: str):
        now = time.time()

        if student_id not in self.active_tracks:
            self.active_tracks[student_id] = now

        duration = now - self.active_tracks[student_id]

        if duration >= self.min_duration:
            self.confirmed.add(student_id)

    def get_confirmed(self):
        return list(self.confirmed)
