# engine/augmentations.py

import cv2
import numpy as np
import random

def apply_augmentations(face):
    augmented = []

    # original
    augmented.append(face)

    # horizontal flip
    augmented.append(cv2.flip(face, 1))

    # brightness & contrast
    for _ in range(2):
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.randint(-20, 20)    # brightness
        aug = cv2.convertScaleAbs(face, alpha=alpha, beta=beta)
        augmented.append(aug)

    # random occlusion (simulate mask / laptop)
    h, w, _ = face.shape
    occ = face.copy()
    x1 = random.randint(0, w // 4)
    y1 = random.randint(h // 2, h - h // 4)
    x2 = x1 + random.randint(w // 4, w // 2)
    y2 = h
    cv2.rectangle(occ, (x1, y1), (x2, y2), (0, 0, 0), -1)
    augmented.append(occ)

    return augmented
