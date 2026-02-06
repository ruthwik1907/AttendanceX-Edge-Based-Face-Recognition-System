import cv2
import numpy as np

class FaceCropper:
    @staticmethod
    def full(face, box):
        x1,y1,x2,y2 = box
        return face[y1:y2, x1:x2]

    @staticmethod
    def upper(face, box):
        x1,y1,x2,y2 = box
        h = y2 - y1
        return face[y1:y1 + int(h * 0.55), x1:x2]
