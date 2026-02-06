from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[1]
STUDENTS_DIR = BASE_DIR / "data" / "students"

def write_attendance(results):
    for student_id, record in results.items():
        student_dir = STUDENTS_DIR / student_id / "attendance"
        student_dir.mkdir(parents=True, exist_ok=True)

        file = student_dir / f"{record['session_id']}.json"
        with open(file, "w") as f:
            json.dump(record, f, indent=2)
