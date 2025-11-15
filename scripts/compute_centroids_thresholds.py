
import numpy as np, json
from pathlib import Path
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import torch

DATA=Path('data/processed'); OUT=Path('models')
device='cuda' if torch.cuda.is_available() else 'cpu'
resnet=InceptionResnetV1(pretrained='vggface2').eval().to(device)
transform=transforms.Compose([transforms.Resize((160,160)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
db={}

for regdir in DATA.iterdir():
    if not regdir.is_dir(): continue
    embs=[]
    for p in regdir.glob('*.jpg'):
        img=Image.open(p).convert('RGB')
        with torch.no_grad():
            emb=resnet(transform(img).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
        embs.append(emb)
    if embs:
        embs=np.vstack(embs); centroid=embs.mean(0)
        dists=np.linalg.norm(embs-centroid,axis=1)
        db[regdir.name]={
            'centroid':centroid.tolist(),
            'mean_dist':float(dists.mean()),
            'std_dist':float(dists.std()),
            'threshold':float(dists.mean()+2*dists.std())
        }

(OUT/'embeddings_db.json').write_text(json.dumps(db,indent=2))
