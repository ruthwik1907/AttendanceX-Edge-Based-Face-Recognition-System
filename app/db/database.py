from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./data/attendancex.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def ensure_schema():
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, student_id TEXT)"))
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(attendance)")).fetchall()]

        if "session_id" not in cols:
            conn.execute(text("ALTER TABLE attendance ADD COLUMN session_id TEXT"))
        if "entry_time" not in cols:
            conn.execute(text("ALTER TABLE attendance ADD COLUMN entry_time DATETIME"))
        if "exit_time" not in cols:
            conn.execute(text("ALTER TABLE attendance ADD COLUMN exit_time DATETIME"))
        if "duration_sec" not in cols:
            conn.execute(text("ALTER TABLE attendance ADD COLUMN duration_sec INTEGER DEFAULT 0"))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS attendance_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                is_active BOOLEAN,
                started_at DATETIME,
                ended_at DATETIME
            )
        """))

        sess_cols = [r[1] for r in conn.execute(text("PRAGMA table_info(attendance_sessions)")).fetchall()]
        if "name" not in sess_cols:
            conn.execute(text("ALTER TABLE attendance_sessions ADD COLUMN name TEXT"))
        if "staff_incharge" not in sess_cols:
            conn.execute(text("ALTER TABLE attendance_sessions ADD COLUMN staff_incharge TEXT"))
        if "class_name" not in sess_cols:
            conn.execute(text("ALTER TABLE attendance_sessions ADD COLUMN class_name TEXT"))
        if "unknown_count" not in sess_cols:
            conn.execute(text("ALTER TABLE attendance_sessions ADD COLUMN unknown_count INTEGER DEFAULT 0"))
        if "thumbnail_path" not in sess_cols:
            conn.execute(text("ALTER TABLE attendance_sessions ADD COLUMN thumbnail_path TEXT"))

        conn.commit()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
