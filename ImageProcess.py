import cv2,os
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
#All these requirements can be installed using the requirements.txt file in the project


DATADIR_PATH= "ImageData" #path of the dataset
DATA_TYPE = os.listdir(DATADIR_PATH)
label_arr = np.identity(len(DATA_TYPE))
IMG_SIZE = 150

"""Function to create training data; images and labels.
This is preprocessing the image, changing its size and selecting single colour"""
def create_data():
    training_data =[]
    count = 0
    for category in DATA_TYPE:
        path = os.path.join(DATADIR_PATH,category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                #Changing image color, RGB to Grey
                resized_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                #changing the size of the image into 150*150
                training_data.append((resized_array,label_arr[count]))
            except Exception as e:
                print('Exception listing images:', e)
        count+=1
    random.shuffle(training_data) #shuffling of data for the training
    return training_data

#############################calling the Function to create data#########################################
train_data1 = create_data()

####################Code for seperately dividing training and testing data from whole data################
TrainTestSplit = 80*(len(train_data1))//100
train1 = np.array([i[0] for i in train_data1[:TrainTestSplit]])/255.0#Dividing by 255 to keep the value of tensor between 0-1
train_data = np.reshape(train1,(-1,IMG_SIZE,IMG_SIZE,1))
test_data1 = np.array([i[0] for i in train_data1[TrainTestSplit:]])/255.0
test_data = np.reshape(test_data1,(-1,IMG_SIZE,IMG_SIZE,1))
train_label = np.array([i[1] for i in train_data1[:TrainTestSplit]])
test_label = np.array([i[1] for i in train_data1[TrainTestSplit:]])

################Code to train model with whole data###################
# train1 = np.array([i[0] for i in train_data1])/255.0
# train_data = np.reshape(train1,(-1,IMG_SIZE,IMG_SIZE,1))
# train_label = np.array([i[1] for i in train_data1])

#####################Creating the model###############################
model=Sequential()
model.add(Conv2D(16,(3,3),input_shape=(IMG_SIZE,IMG_SIZE,1)))#2D convolution layer with 16 nodes
model.add(Activation('relu'))#Rectified Linear unit function
model.add(MaxPooling2D(pool_size=(2,2)))# Selecting maximum element with filter of size 2*2
model.add(Conv2D(16,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3)))#2D convolution layer with 32 nodes
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

######################Callback object for saving the best model based on the 'val_loss' value automatically#########################
######################This will save the model at the locattion of the project######################################################
custom_callback = ModelCheckpoint('model-{epoch:02d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_label,epochs=20,callbacks=[custom_callback],validation_split=0.1)

#################################Model accuracy using the test data################################################################
print(model.evaluate(test_data,test_label))

################################Visualizing the accuracy of the model with each epoch using matplotlib############################
plt.plot(history.history['val_accuracy'],label='Accuracy(Validation)', color='red')
plt.plot(history.history['accuracy'],'r',label='Accuracy(Training)', color='green')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show(block=False)
plt.pause(3)
plt.close()

#######################Visualization of training and validation loss###############################################################
plt.plot(history.history['val_loss'],label='losses(Validation)', color='red')
plt.plot(history.history['loss'],'r',label='losses(Training)', color='green')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show(block=False)
plt.pause(3)
plt.close()

#################################Object Detection using Haar feature-based cascade classifiers#######################################
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0) #'0' in videocapture is for first camera of the machine
while True:
    get,cap_img= cap.read()
    gray = cv2.cvtColor(cap_img,cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in detected_faces:
        f_img = gray[y:y+w,x:x+w]#Changing the color of the image, here we are selecting the face image for prediction
        img_resize = cv2.resize(f_img,(IMG_SIZE,IMG_SIZE))# resizing the image into 150*150 dimension
        normal=img_resize/255.0 #Keeping the range of values in the tensor between 0 to 1
        arr_reshape = np.reshape(normal,(-1,IMG_SIZE,IMG_SIZE,1))#changing shape of the array
        final = model.predict(arr_reshape)#Predicting the type of image using the trained model

        final_category = np.argmax(final,axis=1)[0]

        if (final_category == 0):
            cv2.rectangle(cap_img, (x, y), (x + w, y + h), (204, 204, 0), 4)
        elif (final_category == 1):
            cv2.rectangle(cap_img, (x, y), (x + w, y + h), (0, 0, 255), 4)
        else:
            cv2.rectangle(cap_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.rectangle(cap_img, (x - 10, y - 50), (x + w, y), (105, 105, 105), -1)#This is to set rectangle over the frame
        cv2.putText(cap_img, DATA_TYPE[final_category], (x, y - 15), cv2.FONT_ITALIC, 1, (255, 255, 0), 3)#This is to Put the text in the rectangle

    cv2.imshow('frame', cap_img)
    if cv2.waitKey(1) & 0xFF == ord('q'): #q for quit
        break

cap.release()
cv2.destroyAllWindows()