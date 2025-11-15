
import torch, numpy as np, joblib, sys
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATA = Path('data/processed')
OUT = Path('models'); OUT.mkdir(exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
transform = transforms.Compose([transforms.Resize((160,160)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

X=[]; y=[]
for regdir in sorted([d for d in DATA.iterdir() if d.is_dir()]):
    for p in regdir.glob('*'):
        if p.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.webp'): continue
        img=Image.open(p).convert('RGB')
        with torch.no_grad():
            emb=resnet(transform(img).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
        X.append(emb); y.append(regdir.name)

if not X:
    print("No data to train."); sys.exit(1)

X=np.vstack(X); y=np.array(y)
np.savez(OUT/'train_embeddings.npz',X=X,y=y)
le=LabelEncoder(); y_enc=le.fit_transform(y)
scaler=StandardScaler().fit(X); Xs=scaler.transform(X)

splits=min(5,max(2,len(np.unique(y_enc))))
skf=StratifiedKFold(n_splits=splits,shuffle=True,random_state=42)
scores=[]
for tr,va in skf.split(Xs,y_enc):
    clf=SVC(kernel='linear',probability=True)
    clf.fit(Xs[tr],y_enc[tr])
    scores.append(accuracy_score(y_enc[va], clf.predict(Xs[va])))
print("CV scores:",scores)

clf=SVC(kernel='linear',probability=True)
clf.fit(Xs,y_enc)
joblib.dump({'scaler':scaler,'label_encoder':le,'classifier':clf}, OUT/'face_classifier.joblib')
