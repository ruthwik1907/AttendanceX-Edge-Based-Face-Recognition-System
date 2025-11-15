
import numpy as np, json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

MODELS=Path('models')
data=np.load(MODELS/'train_embeddings.npz')
X=data['X']; y=data['y']

le=LabelEncoder(); y_int=le.fit_transform(y)
num_classes=len(np.unique(y_int))
y_oh=keras.utils.to_categorical(y_int,num_classes)

model=keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(num_classes,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X,y_oh,epochs=25,batch_size=32,validation_split=0.1)

tfl = tf.lite.TFLiteConverter.from_keras_model(model).convert()
(MODELS/'classifier.tflite').write_bytes(tfl)
(MODELS/'label_map.json').write_text(json.dumps({str(i):lab for i,lab in enumerate(le.classes_)}))
