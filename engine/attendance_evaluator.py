# engine/attendance_evaluator.py

class AttendanceEvaluator:
    def __init__(self, min_ratio=0.6):
        """
        min_ratio = % of average class presence time required
        """
        self.min_ratio = min_ratio

    def evaluate(self, presence_tracker):
        times = {
            sid: presence_tracker.get_total_time(sid)
            for sid in presence_tracker.all_students()
        }

        if not times:
            return {}

        avg_time = sum(times.values()) / len(times)

        attendance = {}
        for sid, t in times.items():
            attendance[sid] = {
                "time_present": t,
                "present": t >= self.min_ratio * avg_time
            }

        return attendance
