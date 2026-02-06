from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import AttendanceSession


def is_session_active() -> bool:
    db: Session = SessionLocal()
    try:
        session = (
            db.query(AttendanceSession)
            .filter(AttendanceSession.is_active == True)
            .first()
        )
        return session is not None
    finally:
        db.close()
