
import cv2, torch, json, time, joblib
from pathlib import Path
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

device='cuda' if torch.cuda.is_available() else 'cpu'
mtcnn=MTCNN(image_size=160,margin=20,keep_all=False,device=device)
resnet=InceptionResnetV1(pretrained='vggface2').eval().to(device)

art=joblib.load('models/face_classifier.joblib')
scaler=art['scaler']; le=art['label_encoder']; clf=art['classifier']

transform=transforms.Compose([transforms.Resize((160,160)),transforms.ToTensor(),transforms.Normalize([0.5]*3,[0.5]*3)])

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret: break
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    face=mtcnn(rgb)
    if face is not None:
        with torch.no_grad():
            emb=resnet(face.unsqueeze(0).to(device)).cpu().numpy()
        Xs=scaler.transform(emb)
        probs=clf.predict_proba(Xs)[0]
        idx=int(np.argmax(probs))
        conf=probs[idx]
        reg=le.inverse_transform([idx])[0]
        cv2.putText(frame,f"{reg} ({conf:.2f})",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    cv2.imshow("Live",frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break
cap.release(); cv2.destroyAllWindows()
