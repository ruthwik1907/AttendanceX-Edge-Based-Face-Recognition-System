import sys, os
sys.path.append(os.path.abspath("."))
from app.db.database import SessionLocal
from app.db.models import User
from app.auth.security import hash_password

def create_admin():
    username = input("Enter admin username: ").strip()
    password = input("Enter admin password: ").strip()

    if len(username) < 3 or len(password) < 6:
        print("❌ Username or password too short")
        return

    db = SessionLocal()

    existing = db.query(User).filter(User.username == username).first()
    if existing:
        print("❌ User already exists")
        return

    user = User(
        username=username,
        password_hash=hash_password(password)
    )

    db.add(user)
    db.commit()
    db.close()

    print(f"✅ Admin user '{username}' created successfully")

if __name__ == "__main__":
    create_admin()
