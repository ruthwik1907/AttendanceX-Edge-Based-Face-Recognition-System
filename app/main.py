# app/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.templating import Jinja2Templates

from app.routes.auth import router as auth_router
from app.routes.enroll import router as enroll_router
from app.routes.students import router as students_router
from app.routes.attendance import router as attendance_router
from app.routes.session import router as session_router
from app.routes.student_profile import router as student_profile_router
from app.routes.metrics import router as metrics_router
from app.routes.video import router as video_router
from app.routes.users import router as users_router
from app.db.database import ensure_schema

app = FastAPI(
    title="AttendanceX",
    docs_url=None,
    redoc_url=None
)

# Session middleware (login)
app.add_middleware(
    SessionMiddleware,
    secret_key="attendancex-secret-key"
)

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Routes
app.include_router(auth_router)
app.include_router(enroll_router)
app.include_router(students_router)
app.include_router(attendance_router)
app.include_router(session_router)
app.include_router(student_profile_router)
app.include_router(metrics_router)
app.include_router(video_router)
app.include_router(users_router)

templates = Jinja2Templates(directory="app/templates")

@app.get("/")
def root():
    return {"status": "AttendanceX running"}

@app.exception_handler(StarletteHTTPException)
def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 303:
        return RedirectResponse(exc.headers.get("Location", "/login"), status_code=303)
    if exc.status_code >= 400:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "status_code": exc.status_code,
                "title": "Something went wrong",
                "message": str(exc.detail) if exc.detail else "An unexpected error occurred."
            },
            status_code=exc.status_code
        )
    return HTMLResponse(str(exc.detail), status_code=exc.status_code)

@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": 500,
            "title": "Internal Server Error",
            "message": "The server encountered an error. Please try again."
        },
        status_code=500
    )

@app.on_event("startup")
def on_startup():
    ensure_schema()
