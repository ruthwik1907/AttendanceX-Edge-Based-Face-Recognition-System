# engine/pipeline_live.py

import cv2

from engine.detector_yolo import YOLOFaceDetector
from engine.tracker import FaceTracker
from engine.utils import FPSCounter
from engine.recognizer_arcface import ArcFaceRecognizer
from engine.embedding_store import EmbeddingStore
from engine.track_identity import TrackIdentityManager
from engine.presence_tracker import PresenceTracker
from engine.attendance_evaluator import AttendanceEvaluator
from engine.session_client import is_session_active
from engine.session_state import is_session_active
import time




# -----------------------------
# INITIALIZATION
# -----------------------------
detector = YOLOFaceDetector(
    model_path="models/yolov8n-face.pt",
    conf=0.4
)

tracker = FaceTracker()
fps_counter = FPSCounter()

recognizer = ArcFaceRecognizer("models/arcface.onnx")

store = EmbeddingStore(threshold=0.6)
store.load_from_disk("data/students")

identity_mgr = TrackIdentityManager(
    cooldown=1.5,
    min_frames=3
)

presence_tracker = PresenceTracker()
attendance_eval = AttendanceEvaluator(min_ratio=0.6)


# -----------------------------
# MAIN
# -----------------------------
def main():
    cap = None
    camera_on = False

    print("🟡 Pipeline started — waiting for session")

    while True:
        session_active = is_session_active()

        # -------------------------
        # SESSION JUST STARTED
        # -------------------------
        if session_active and not camera_on:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Camera not accessible")
                time.sleep(2)
                continue

            camera_on = True
            print("🎥 Camera activated (session started)")

        # -------------------------
        # SESSION JUST STOPPED
        # -------------------------
        if not session_active and camera_on:
            cap.release()
            camera_on = False
            print("🛑 Camera released (session stopped)")
            time.sleep(1)
            continue

        # -------------------------
        # IDLE STATE
        # -------------------------
        if not camera_on:
            time.sleep(1)
            continue

        # -------------------------
        # ACTIVE PIPELINE
        # -------------------------
        ret, frame = cap.read()
        if not ret:
            continue

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        fps = fps_counter.update()

        active_track_ids = set()

        for x1, y1, x2, y2, track_id in tracks:
            active_track_ids.add(track_id)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            identity = identity_mgr.get_identity(track_id)

            if identity is None:
                embedding = recognizer.get_embedding(face)
                if embedding is not None:
                    student_id, score = store.match(embedding)
                    if student_id:
                        identity_mgr.update(track_id, student_id, score)

            label = identity_mgr.get_identity(track_id)
            if label:
                presence_tracker.update(label)

            display = label if label else f"Unknown {track_id}"
            color = (0,255,0) if label else (0,200,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame, display,(x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        identity_mgr.cleanup(active_track_ids)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

        cv2.imshow("AttendanceX — Live", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # -----------------------------
    # SESSION END → ATTENDANCE
    # -----------------------------
    
    if cap:
        cap.release()
    cv2.destroyAllWindows()




    attendance = attendance_eval.evaluate(presence_tracker)

    print("\n📋 FINAL ATTENDANCE REPORT")
    for sid, info in attendance.items():
        status = "PRESENT" if info["present"] else "ABSENT"
        print(f"{sid}: {status} ({info['time_present']:.1f}s)")


if __name__ == "__main__":
    main()
