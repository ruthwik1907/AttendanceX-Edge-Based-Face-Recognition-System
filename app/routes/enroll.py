from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import json
from datetime import datetime
from shutil import rmtree
from app.db.database import SessionLocal
from app.db.models import Student



from app.auth.dependencies import require_login
from engine.registration import RegistrationEngine

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

BASE_DIR = Path(__file__).resolve().parents[2]
STUDENTS_DIR = BASE_DIR / "data" / "students"
STUDENTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize engine ONCE
engine = RegistrationEngine(
    detector_model="models/yolov8n-face.pt",
    recognizer_model="models/arcface.onnx"
)


@router.get("/enroll", response_class=HTMLResponse)
def enroll_page(request: Request):
    require_login(request)
    return templates.TemplateResponse("enroll.html", {"request": request})


@router.post("/enroll")
async def enroll_student(
    request: Request,
    reg: str = Form(...),
    files: list[UploadFile] = File(...)
):
    require_login(request)

    if not reg.isalnum():
        raise HTTPException(status_code=400, detail="Invalid student ID")

    if len(files) < 10:
        raise HTTPException(status_code=400, detail="At least 10 face images required")

    student_dir = STUDENTS_DIR / reg
    samples_dir = student_dir / "samples"
    attendance_dir = student_dir / "attendance"

    is_update = student_dir.exists()

    if is_update:
        embeddings_file = student_dir / "embeddings.npy"
        if samples_dir.exists():
            rmtree(samples_dir)
        if embeddings_file.exists():
            embeddings_file.unlink()
    else:
        student_dir.mkdir(parents=True, exist_ok=True)
        attendance_dir.mkdir(exist_ok=True)

    # recreate samples directory safely
    samples_dir.mkdir(parents=True, exist_ok=True)

    # save images
    for i, file in enumerate(files):
        img_path = samples_dir / f"img_{i:03d}.jpeg"
        with open(img_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    # run registration
    image_list = sorted(samples_dir.glob("*.jpeg"))

    record = engine.register_student(
        student_id=reg,
        name=reg,
        images=image_list
    )

    meta = {
        "student_id": reg,
        "samples_used": record["samples_used"],
        "last_updated": datetime.now().isoformat(),
        "updated": is_update
    }
    db = SessionLocal()

    existing = db.query(Student).filter(
        Student.student_id == reg
    ).first()

    if existing:
        existing.samples_used = record["samples_used"]
        existing.created_at = datetime.utcnow()
    else:
        student = Student(
            student_id=reg,
            name=reg,
            samples_used=record["samples_used"]
        )
        db.add(student)

    db.commit()
    db.close()


    with open(student_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "status": "success",
        "student_id": reg,
        "updated": is_update,
        "samples_used": record["samples_used"]
    }

