
# FULL combined processing: autorotate → detect → align → collect failed → augment → report
from pathlib import Path
import shutil, time, random, json
from facenet_pytorch import MTCNN
from PIL import Image, ExifTags, ImageEnhance, ImageFilter
import torch, numpy as np

ROOT = Path(".")
ENROLL = ROOT/"data/enroll"
PROCESSED = ROOT/"data/processed"
FAILED = ROOT/"data/failed_detection"
REPORTS = ROOT/"reports"

for d in [PROCESSED, FAILED, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tiff"}
MIN_REQUIRED = 30

def autorotate(im):
    try:
        for orientation in ExifTags.TAGS:
            if ExifTags.TAGS[orientation]=="Orientation":
                exif = im._getexif()
                if exif is None: return im
                orient = exif.get(orientation,None)
                if orient==3: return im.rotate(180,expand=True)
                if orient==6: return im.rotate(270,expand=True)
                if orient==8: return im.rotate(90,expand=True)
    except: pass
    return im

def augment_once(img):
    funcs=[]
    def rrotate(img): return img.rotate(random.uniform(-15,15), resample=Image.BICUBIC, expand=False)
    def rbc(img): a=ImageEnhance.Brightness(img).enhance(random.uniform(0.8,1.2)); return ImageEnhance.Contrast(a).enhance(random.uniform(0.85,1.15))
    def rnoise(img): arr=np.array(img).astype("float32"); var=random.uniform(5,50); noise=np.random.normal(0,var**0.5, arr.shape); return Image.fromarray(np.clip(arr+noise,0,255).astype("uint8"))
    def rblur(img): 
        if random.random()<0.4: return img.filter(ImageFilter.GaussianBlur(radius=random.choice([1,2,3])))
        return img
    def rdrop(img):
        arr=np.array(img); h,w=arr.shape[:2]
        for _ in range(random.randint(1,3)):
            hh=random.randint(max(1,int(0.02*h)),max(1,int(0.12*h)))
            ww=random.randint(max(1,int(0.02*w)),max(1,int(0.12*w)))
            y=random.randint(0,max(0,h-hh)); x=random.randint(0,max(0,w-ww))
            arr[y:y+hh,x:x+ww]=np.random.randint(0,255,(hh,ww,3),dtype="uint8")
        return Image.fromarray(arr)
    funcs=[rrotate,rbc,rnoise,rblur,rdrop]
    out=img
    for f in random.sample(funcs, random.randint(1,3)):
        out=f(out)
    return out

report=[]
for regdir in sorted([d for d in ENROLL.iterdir() if d.is_dir()]):
    reg=regdir.name
    files=[p for p in regdir.iterdir() if p.suffix.lower() in IMG_EXTS]
    dest = PROCESSED/reg
    dest.mkdir(exist_ok=True)
    # Skip if already processed
    if sum(1 for _ in dest.iterdir()) >= MIN_REQUIRED:
        report.append({"RegistrationID":reg, "Skipped":True})
        continue
    failed=[]
    passed=[]
    for p in files:
        try:
            im=autorotate(Image.open(p).convert("RGB"))
            boxes,_=mtcnn.detect(np.array(im))
            if boxes is None:
                failed.append(p.name); (FAILED/reg).mkdir(exist_ok=True,parents=True); shutil.copy2(p, FAILED/reg/p.name)
            else:
                face=mtcnn(im)
                if face is None:
                    failed.append(p.name); (FAILED/reg).mkdir(exist_ok=True,parents=True); shutil.copy2(p, FAILED/reg/p.name)
                else:
                    arr=((face.permute(1,2,0).numpy()+1)/2*255).astype("uint8")
                    Image.fromarray(arr).save(dest/p.name)
                    passed.append(p.name)
        except:
            failed.append(p.name); (FAILED/reg).mkdir(exist_ok=True,parents=True); shutil.copy2(p,FAILED/reg/p.name)
    # Augment
    for fp in sorted(dest.iterdir()):
        if fp.suffix.lower() in IMG_EXTS:
            img=Image.open(fp).convert("RGB")
            for i in range(2):
                aug=augment_once(img)
                aug.save(dest/f"{fp.stem}_aug{i}.jpg")
    report.append({
        "RegistrationID": reg,
        "Uploaded": len(files),
        "Passed": len(passed),
        "Failed": len(failed),
        "Skipped": False
    })

import csv

if not report:
    print("No report rows to write.")
else:
    # compute union of all keys across report rows so CSV header contains every possible column
    fieldnames_set = set()
    for r in report:
        if isinstance(r, dict):
            fieldnames_set.update(r.keys())

    # prefer a stable, readable order: common fields first, then the rest sorted
    preferred_order = ["RegistrationID", "Uploaded", "Passed", "Failed", "Skipped"]
    # include any other fields that appeared
    other_fields = [f for f in sorted(fieldnames_set) if f not in preferred_order]
    fieldnames = [f for f in preferred_order if f in fieldnames_set] + other_fields

    out_path = REPORTS / "failed_detection_report.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        # ensure every row is a dict and write rows in the same column order
        for row in report:
            if not isinstance(row, dict):
                # skip bad entries (shouldn't happen) but keep log
                continue
            # create a canonical row mapping missing fields -> ''
            canonical = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(canonical)

    print(f"Wrote report CSV: {out_path}")


print("process_all completed.")
