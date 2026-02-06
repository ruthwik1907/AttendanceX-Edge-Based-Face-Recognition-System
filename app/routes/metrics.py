from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from app.auth.dependencies import require_login
from engine.metrics import Metrics

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/metrics")
def metrics(request: Request):
    require_login(request)
    return templates.TemplateResponse(
        "metrics.html",
        {
            "request": request,
            "fps": round(Metrics.fps(), 2),
            "frames": Metrics.frames,
            "detections": Metrics.detections
        }
    )

@router.get("/metrics/json")
def metrics_json(request: Request):
    require_login(request)
    return {
        "fps": round(Metrics.fps(), 2),
        "frames": Metrics.frames,
        "detections": Metrics.detections
    }
