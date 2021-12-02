import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import callbacks

from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics


les_fichiers=glob.glob("dataset/*")
my_dico={}
my_dico["filepath"]=les_fichiers
liste_name=[]

results = open("reports.txt", "a")

for fichier in les_fichiers:   
    name=re.search(r"[a-zA-Z*_?a-zA-Z*]*", os.path.basename(fichier))
    liste_name.append(name.group(0))
    my_dico["label"]=liste_name
    
df=pd.DataFrame.from_dict(my_dico)

df["label"]=df["label"].replace({"Alfred_Sisley_": 0, "Frida_Kahlo_": 1,"Andrei_Rublev_":2,"Gustave_Courbet_":3 })

X_train_path,X_test_path,y_train,y_test=train_test_split(df["filepath"],df["label"],test_size=0.2,random_state=1234)
X_test=[]
for filepath in tqdm(X_test_path):
    im=tf.io.read_file(filepath)
    im=tf.image.decode_jpeg(im,channels=3)
    im=tf.image.resize(im,size=(256,256))
    X_test.append([im])

X_test=tf.concat(X_test,axis=0)

def load_image(filepath):
    im=tf.io.read_file(filepath)
    im=tf.image.decode_jpeg(im,channels=3)
    im=tf.image.resize(im,size=(256,256))
    return im

dataset_train=tf.data.Dataset.from_tensor_slices((X_train_path,y_train))
dataset_train=dataset_train.map(lambda x,y:[load_image(x),y],num_parallel_calls=-1).batch(32)

efficientNet=EfficientNetB1(include_top=False,input_shape=(256,256,3))
for layer in efficientNet.layers:
    layer.trainable=False
    
model=Sequential()
model.add(efficientNet)
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4,activation="softmax"))
model.summary()
model.summary(print_fn=lambda x: results.write(x + '\n'))


model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

checkpoint=callbacks.ModelCheckpoint(filepath="checkpoint",monitor="val_loss",save_best_only=True,save_weights_only=False,mode="min",save_freq='epoch')
lr_plateau=callbacks.ReduceLROnPlateau(monitor="val_loss",patience=5,factor=0.1,verbose=2,mode="min")
early_stopping = callbacks.EarlyStopping(monitor = "val_loss",
                                         patience = 2,
                                         mode = 'min',
                                         verbose = 2,
                                         restore_best_weights= True)

   
history = model.fit(dataset_train, 
                              epochs = 10,
                              validation_data = (X_test,y_test),
                              callbacks=[lr_plateau, early_stopping, checkpoint])
    
    
y_prob=model.predict(X_test,batch_size=64)
y_pred=tf.argmax(y_prob,axis=-1).numpy()
results.write(str(accuracy_score(y_test,y_pred)))
results.write(str(confusion_matrix(y_test,y_pred)))
print(str(accuracy_score(y_test,y_pred)))
print(str(confusion_matrix(y_test,y_pred)))

cross_table = pd.crosstab(y_test, y_pred, rownames=['rééles'], colnames=['Prédites'], margins=True)
results.write(str(cross_table))
print(str(cross_table))

cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1,2,3])
disp.plot()
plt.savefig('confusion_matrix.png')


results.write(metrics.classification_report(y_test, y_pred))
results.close()

plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss by epoch')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.subplot(122)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy by epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')
plt.savefig('accuracy_loss.png')
