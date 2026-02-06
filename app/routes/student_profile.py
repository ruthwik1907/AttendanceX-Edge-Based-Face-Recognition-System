from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from app.db.database import SessionLocal
from app.db.models import Student, AttendanceRecord
from app.auth.dependencies import require_login

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/students/{student_id}")
def student_profile(request: Request, student_id: str):
    require_login(request)
    db = SessionLocal()
    student = db.query(Student).filter_by(student_id=student_id).first()
    records = db.query(AttendanceRecord).filter_by(student_id=student_id).all()
    db.close()

    return templates.TemplateResponse(
        "student_profile.html",
        {"request": request, "student": student, "records": records}
    )
