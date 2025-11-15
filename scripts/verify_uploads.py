
import json
from pathlib import Path
import pandas as pd

DATA=Path('data/processed')
OUT=Path('reports'); OUT.mkdir(exist_ok=True)

rows=[]
for d in sorted(DATA.iterdir()):
    if d.is_dir():
        cnt=sum(1 for f in d.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))
        rows.append({'RegistrationID':d.name,'ProcessedCount':cnt})
df=pd.DataFrame(rows)
df.to_csv(OUT/'processed_counts.csv',index=False)

report={'insufficient':[r for r in rows if r['ProcessedCount']<30]}
(OUT/'upload_verification.json').write_text(json.dumps(report,indent=2))
