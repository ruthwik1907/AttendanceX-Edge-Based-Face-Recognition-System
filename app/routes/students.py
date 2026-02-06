from fastapi import APIRouter, Request, Query
from fastapi.templating import Jinja2Templates
from app.db.database import SessionLocal
from app.db.models import Student
from app.auth.dependencies import require_login
from sqlalchemy import func

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/students")
def students_page(
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=5, le=100),
    q: str = Query("", min_length=0),
):
    require_login(request)
    db = SessionLocal()

    base = db.query(Student)
    if q:
        base = base.filter(Student.student_id.contains(q))

    total = base.count()
    total_pages = max(1, (total + size - 1) // size)
    page = min(page, total_pages)

    students = base.order_by(Student.created_at.desc()).offset((page-1)*size).limit(size).all()
    db.close()

    return templates.TemplateResponse(
        "students.html",
        {
            "request": request,
            "students": students,
            "page": page,
            "size": size,
            "q": q,
            "total_pages": total_pages,
            "total": total
        }
    )
