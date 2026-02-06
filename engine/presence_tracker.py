# engine/presence_tracker.py

import time
from collections import defaultdict

class PresenceTracker:
    def __init__(self):
        self.presence = defaultdict(list)

    def update(self, student_id):
        self.presence[student_id].append(time.time())

    def get_total_time(self, student_id):
        times = self.presence.get(student_id, [])
        if len(times) < 2:
            return 0.0
        return times[-1] - times[0]

    def all_students(self):
        return self.presence.keys()
