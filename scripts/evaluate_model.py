
import numpy as np, joblib, json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

MODELS=Path('models')
npz=np.load(MODELS/'train_embeddings.npz')
X=npz['X']; y=npz['y']
art=joblib.load(MODELS/'face_classifier.joblib')
scaler=art['scaler']; le=art['label_encoder']; clf=art['classifier']

Xs=scaler.transform(X)
y_int=le.transform(y)
yp=clf.predict(Xs)
acc=accuracy_score(y_int,yp)

report=classification_report(y_int,yp,target_names=list(le.classes_),output_dict=True)
cm=confusion_matrix(y_int,yp)

REPORTS=Path('reports'); REPORTS.mkdir(exist_ok=True)
(REPORTS/'evaluation_summary.json').write_text(json.dumps({'accuracy':float(acc),'report':report},indent=2))
pd.DataFrame(cm).to_csv(REPORTS/'confusion_matrix.csv',index=False)
print("Accuracy:",acc)
