# scripts/enhance_zero_dce.py
"""
Apply Zero-DCE low-light enhancement on images in a folder.
This is a skeleton that uses a simple gamma fallback if Zero-DCE not installed.
"""
import argparse
from pathlib import Path
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--in', dest='indir', default='data/raw/lowlight')
parser.add_argument('--out', default='data/processed_lowlight_enhanced')
args = parser.parse_args()
IN=Path(args.indir); OUT=Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
for p in IN.rglob('*'):
    if p.suffix.lower() not in ('.jpg','.png','.jpeg'): continue
    img=cv2.imread(str(p))
    if img is None: continue
    # simple gamma correct fallback
    gamma=1.6
    look = (img/255.0) ** (1.0/gamma)
    out = (look*255).astype('uint8')
    cv2.imwrite(str(OUT/p.name), out)
print('Enhanced images written to', OUT)
