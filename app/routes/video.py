# app/routes/video.py

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from engine.live_stream import live_stream
from app.auth.dependencies import require_login

router = APIRouter(prefix="/video")

@router.get("/feed")
def video_feed(request: Request):
    require_login(request)
    return StreamingResponse(
        live_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
