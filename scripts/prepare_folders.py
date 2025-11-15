
import pandas as pd
from pathlib import Path

EXCEL = 'studentMaster.xlsx'
OUT = Path('data/enroll')

df = pd.read_excel(EXCEL, dtype=str)
OUT.mkdir(parents=True, exist_ok=True)

for _, row in df.iterrows():
    reg = str(row['Registration ID']).strip()
    d = OUT / reg
    d.mkdir(exist_ok=True)
    (d / "meta.txt").write_text(f"RegID: {reg}\nName: {row.get('Student Name','')}\nProgramme: {row.get('Programme Name','')}")
print("Folders prepared.")
