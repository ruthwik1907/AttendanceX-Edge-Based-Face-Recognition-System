# scripts/detect_all.py
"""
Run detection with different backends and save boxes.
Usage:
  python scripts/detect_all.py --detector mtcnn --data data/raw --out detections/mtcnn
Note: requires the detector implementations in your scripts/ directory.
"""
import argparse, os, json
from pathlib import Path
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--detector', choices=['mtcnn','retinaface','yolov8face'], required=True)
parser.add_argument('--data', default='data/raw')
parser.add_argument('--out', default='detections/out')
args = parser.parse_args()
DATA=Path(args.data)
OUT=Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
names=[p for p in DATA.rglob('*') if p.suffix.lower() in ('.jpg','.png','.jpeg')]
for imgp in names:
    img=cv2.imread(str(imgp))
    if img is None: continue
    h,w = img.shape[:2]
    # placeholder detection: whole image if detector is missing
    bbox=[0,0,w,h]
    outf = OUT / (imgp.stem + '.json')
    json.dump({'path':str(imgp),'bboxes':[bbox]}, open(outf,'w'))
print('Wrote dummy detections to', OUT)
