import cv2, os, random
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import *


DATADIR_PATH = "C:/Users/HP/Desktop/DataforMask/archive/train"
CATEGORIES = ["incorrect_weared_mask", "with_mask", "without_mask"]
IMG_SIZE = 150
training_data=[]

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR_PATH,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)
X=[]
Y=[]

for features, label in training_data:
    X.append(features)
    Y.append(label)

X= np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X=X/255
Y=np.array(Y)

TrainTestSplit = 80*(len(X))//100
Train_X =  X[0:TrainTestSplit]
Test_X = X[TrainTestSplit:]
Train_Y =  Y[0:TrainTestSplit]
Test_Y = Y[TrainTestSplit:]


model = Sequential()
model.add(Flatten(input_shape=(IMG_SIZE, IMG_SIZE,1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(3,activation='linear'))
#model.add(Softmax())
#model.compile(loss='mse', optimizer='adam')
model.compile(optimizer='adam',loss= SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(X,Y,epochs=20,shuffle=True,verbose=1)


face_cascade = cv2.CascadeClassifier(r'C:\Users\HP\PycharmProjects\maskInPlace\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while True:
    ret,img= cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        f_img = gray[y:y+w,x:x+w]
        img_resize = cv2.resize(f_img,(150,150))
        normal=img_resize/255.0
        arr_reshape = np.reshape(normal,(-1,150,150,1))
        final = model.predict(arr_reshape)

        final_category = np.argmax(final,axis=1)[0]

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0) ,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),(0,0,255) ,-1)
        cv2.putText(img,CATEGORIES[label],(x,y-10),cv2.FONT_ITALIC,0.8,(255,255,255),2)

    cv2.imshow('frame2', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
