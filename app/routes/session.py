from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from app.auth.dependencies import require_login
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path
import cv2

from app.db.database import get_db
from app.db.models import AttendanceSession, AttendanceRecord, Student
from engine.runtime import runtime
from engine.metrics import Metrics

router = APIRouter(prefix="/session")
templates = Jinja2Templates(directory="app/templates")

@router.get("")
def session_page(request: Request, db: Session = Depends(get_db)):
    require_login(request)
    total_students = db.query(Student).count()
    total_records = db.query(AttendanceRecord).count()
    active_session = runtime.running
    return templates.TemplateResponse(
        "session.html",
        {
            "request": request,
            "active": active_session,
            "total_students": total_students,
            "total_records": total_records,
            "fps": round(Metrics.fps(), 2),
            "frames": Metrics.frames,
            "detections": Metrics.detections
        }
    )

@router.get("/status")
def session_status(request: Request):
    require_login(request)
    return {"active": runtime.running}

@router.post("/start")
def start_session(request: Request, db=Depends(get_db)):
    require_login(request)

    db.query(AttendanceSession).update({"is_active": False})
    new_sess = AttendanceSession(is_active=True, started_at=datetime.utcnow())
    db.add(new_sess)
    db.commit()

    runtime.presence_tracker.presence.clear()
    runtime.start()
    return {"status": "started"}

@router.post("/stop")
def stop_session(request: Request, db=Depends(get_db)):
    require_login(request)

    active = db.query(AttendanceSession).filter_by(is_active=True).order_by(AttendanceSession.id.desc()).first()
    if active:
        active.is_active = False
        active.ended_at = datetime.utcnow()

    runtime.stop()

    # Save thumbnail + unknown count
    if active:
        active.unknown_count = len(runtime.unknown_ids)

        if runtime.last_frame is not None:
            out_dir = Path("app/static/sessions")
            out_dir.mkdir(parents=True, exist_ok=True)
            file_path = out_dir / f"session_{active.id}.jpg"
            cv2.imwrite(str(file_path), runtime.last_frame)
            active.thumbnail_path = f"/static/sessions/session_{active.id}.jpg"

    # Write attendance records
    results = runtime.attendance_eval.evaluate(runtime.presence_tracker)
    for sid, info in results.items():
        times = runtime.presence_tracker.presence.get(sid, [])
        if len(times) >= 2:
            entry = datetime.fromtimestamp(times[0])
            exit = datetime.fromtimestamp(times[-1])
            duration = int(times[-1] - times[0])
        else:
            entry = None
            exit = None
            duration = 0

        db.add(AttendanceRecord(
            student_id=sid,
            session_id=str(active.id) if active else None,
            entry_time=entry,
            exit_time=exit,
            duration_sec=duration
        ))

    db.commit()
    return {"status": "stopped"}
