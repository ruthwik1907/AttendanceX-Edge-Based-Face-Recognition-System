from datetime import datetime

class SessionController:
    def __init__(self):
        self.active = False

    def start(self):
        self.active = True
        self.started_at = datetime.now()

    def stop(self):
        self.active = False
        self.ended_at = datetime.now()
