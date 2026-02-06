# scripts/init_db.py
import sys, os
sys.path.append(os.path.abspath("."))
from app.db.database import engine
from app.db.models import Base

Base.metadata.create_all(bind=engine)
print("✅ Database initialized")
