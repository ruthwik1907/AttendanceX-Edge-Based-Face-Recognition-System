from fastapi import APIRouter, Request, Form, Query
from fastapi.templating import Jinja2Templates
from app.db.database import SessionLocal
from app.db.models import AttendanceRecord, AttendanceSession
from app.auth.dependencies import require_login
from sqlalchemy import func, distinct
from fastapi.responses import RedirectResponse

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/attendance")
def attendance_page(
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(6, ge=5, le=50),
    q: str = Query("", min_length=0)
):
    require_login(request)
    db = SessionLocal()

    base = db.query(AttendanceSession)
    if q:
        base = base.filter(
            (AttendanceSession.name.contains(q)) |
            (AttendanceSession.class_name.contains(q)) |
            (AttendanceSession.staff_incharge.contains(q))
        )

    total = base.count()
    total_pages = max(1, (total + size - 1) // size)
    page = min(page, total_pages)

    sessions = base.order_by(AttendanceSession.started_at.desc()).offset((page-1)*size).limit(size).all()

    counts = dict(
        db.query(
            AttendanceRecord.session_id,
            func.count(distinct(AttendanceRecord.student_id))
        ).group_by(AttendanceRecord.session_id).all()
    )

    db.close()

    return templates.TemplateResponse(
        "attendance.html",
        {
            "request": request,
            "sessions": sessions,
            "counts": counts,
            "page": page,
            "size": size,
            "q": q,
            "total_pages": total_pages,
            "total": total
        }
    )

@router.post("/attendance/session/{session_id}/edit")
def edit_session(
    request: Request,
    session_id: int,
    name: str = Form(None),
    staff_incharge: str = Form(None),
    class_name: str = Form(None),
):
    require_login(request)
    db = SessionLocal()
    sess = db.query(AttendanceSession).filter_by(id=session_id).first()
    if sess:
        sess.name = name
        sess.staff_incharge = staff_incharge
        sess.class_name = class_name
        db.commit()
    db.close()
    return RedirectResponse("/attendance", status_code=303)

@router.get("/attendance/session/{session_id}")
def session_detail(
    request: Request,
    session_id: int,
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=5, le=100),
    q: str = Query("", min_length=0)
):
    require_login(request)
    db = SessionLocal()

    sess = db.query(AttendanceSession).filter_by(id=session_id).first()
    base = db.query(AttendanceRecord).filter_by(session_id=str(session_id))
    if q:
        base = base.filter(AttendanceRecord.student_id.contains(q))

    total = base.count()
    total_pages = max(1, (total + size - 1) // size)
    page = min(page, total_pages)

    records = base.order_by(AttendanceRecord.entry_time.desc()).offset((page-1)*size).limit(size).all()
    db.close()

    return templates.TemplateResponse(
        "session_detail.html",
        {
            "request": request,
            "session": sess,
            "records": records,
            "page": page,
            "size": size,
            "q": q,
            "total_pages": total_pages
        }
    )
