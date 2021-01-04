# Intelligent-Identifier

Intelligent identifier is an interesting project which classify an image captured by camera based on mask on person's face into three categories i.e 
1. Incorrectly wore
2. With mask
3. Without mask

In this project I built a sequential model with series of convolution neural network in Keras with python using dataset([Credits:vinay kudari](https://www.kaggle.com/vinaykudari/facemask)) available on kaggle.
This project is carried through using OpenCV, Keras, Tensorflow.

Procedure:

1). In the first step the images in the dataset are processed to change its colour and size using openCV library. Labelling of the dataset is also done based on their      category. Data is reshufelled for training the model.<br /> 
2). Data is divided into training set and test set and converted into arrays and reshaped using NumPy library. Here I divided whole data, 80% for the training of model and 20% for testing.<br />
3). Model based on sequential stack is made with four 2-dimensional convolution layer of different nodes. Rectified linear unit function along with downsampling of the input representation by maxpooling is done after these four layers. Then input is flattened and dropout layer is used to prevent overfitting. Two dense layer with 64 and 3 neurons are deployed.<br />
4). Model is trained with validation split of 0.1 and various losses and accuracy are observed using Pyplot module.<br />
5). Camera is used to capture images, Haar feature-based cascade classifier is used to detect face in the image, the faced image is cropped, arrayed, resized and reshaped and fed to model for the prediction.<br />
6). Based on the prediction, rectangle with different text upon it is sketched around the face.<br />

Result:<br />
Classification model which can proficiently detects the type of input images with an accuracy upto 81%.
