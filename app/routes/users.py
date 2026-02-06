from fastapi import APIRouter, Request, Form, Query
from fastapi.templating import Jinja2Templates
from app.db.database import SessionLocal
from app.db.models import User
from app.auth.dependencies import require_login
from app.auth.security import hash_password

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/users")
def users_page(
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=5, le=100),
    q: str = Query("", min_length=0)
):
    require_login(request)
    db = SessionLocal()

    base = db.query(User)
    if q:
        base = base.filter(User.username.contains(q))

    total = base.count()
    total_pages = max(1, (total + size - 1) // size)
    page = min(page, total_pages)

    users = base.order_by(User.username.asc()).offset((page-1)*size).limit(size).all()
    db.close()

    return templates.TemplateResponse(
        "users.html",
        {"request": request, "users": users, "page": page, "size": size, "q": q, "total_pages": total_pages}
    )

@router.post("/users")
def add_user(request: Request, username: str = Form(...), password: str = Form(...)):
    require_login(request)
    db = SessionLocal()
    exists = db.query(User).filter_by(username=username).first()
    if exists:
        db.close()
        return templates.TemplateResponse(
            "users.html",
            {"request": request, "users": db.query(User).all(), "error": "User already exists"},
            status_code=400
        )
    db.add(User(username=username, password_hash=hash_password(password)))
    db.commit()
    db.close()
    return RedirectResponse("/users", status_code=303)

@router.post("/users/delete")
def delete_user(request: Request, username: str = Form(...)):
    require_login(request)
    current = request.session.get("user")
    if username == current:
        return RedirectResponse("/users", status_code=303)

    db = SessionLocal()
    db.query(User).filter_by(username=username).delete()
    db.commit()
    db.close()
    return RedirectResponse("/users", status_code=303)