# engine/phase3/types.py
from dataclasses import dataclass
from typing import Tuple
import time

@dataclass
class TrackedFace:
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float
