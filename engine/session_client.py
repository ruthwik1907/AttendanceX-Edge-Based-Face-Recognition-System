import requests

def is_session_active():
    try:
        r = requests.get("http://127.0.0.1:8000/session/status")
        return r.json()["active"]
    except:
        return False
