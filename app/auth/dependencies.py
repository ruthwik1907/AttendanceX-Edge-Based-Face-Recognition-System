from fastapi import Request, HTTPException

def require_login(request: Request):
    if not request.session.get("user"):
        raise HTTPException(status_code=303, headers={"Location": "/login"})
