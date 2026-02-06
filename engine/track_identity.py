import time

class TrackIdentityManager:
    def __init__(self, cooldown=2.0, min_frames=3):
        """
        cooldown   : seconds before a lost track is removed
        min_frames : consecutive recognitions before identity is locked
        """
        self.tracks = {}
        self.cooldown = cooldown
        self.min_frames = min_frames

    def update(self, track_id, student_id, score):
        now = time.time()

        state = self.tracks.get(track_id)
        if state is None:
            state = {
                "student_id": None,
                "hits": 0,
                "confidence": 0.0,
                "last_seen": now,
            }

        if student_id:
            state["hits"] += 1
            state["confidence"] = score

            if state["hits"] >= self.min_frames:
                state["student_id"] = student_id

        state["last_seen"] = now
        self.tracks[track_id] = state

    def get_identity(self, track_id):
        state = self.tracks.get(track_id)
        if not state:
            return None

        if state["student_id"]:
            return f"{state['student_id']} ({state['confidence']:.2f})"

        return None

    def cleanup(self, active_track_ids):
        """
        Remove tracks that disappeared longer than cooldown
        """
        now = time.time()
        to_remove = []

        for tid, state in self.tracks.items():
            if tid not in active_track_ids:
                if now - state["last_seen"] > self.cooldown:
                    to_remove.append(tid)

        for tid in to_remove:
            del self.tracks[tid]
