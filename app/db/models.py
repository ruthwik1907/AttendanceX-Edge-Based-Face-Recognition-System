from sqlalchemy import Boolean, Column, ForeignKey, String, Integer, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    password_hash = Column(String)

class Student(Base):
    __tablename__ = "students"

    student_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    samples_used = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

class Session(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, default=datetime.utcnow)

class AttendanceRecord(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey("students.student_id"))
    session_id = Column(String, nullable=True)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    duration_sec = Column(Integer, default=0)

class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    is_active = Column(Boolean, default=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    name = Column(String, nullable=True)
    staff_incharge = Column(String, nullable=True)
    class_name = Column(String, nullable=True)
    unknown_count = Column(Integer, default=0)
    thumbnail_path = Column(String, nullable=True)
