# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name:PREETHA.S
### Register Number:212222230110
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

X_train.shape

X_test.shape
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
print("PREETHA S, 212222230110")
single_image = X_train[400]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
print("Preetha S, 212222230110")
metrics.head()
print("Preetha S, 212222230110")
metrics[['accuracy','val_accuracy']].plot()
print("Preetha S, 212222230110")
metrics[['loss','val_loss']].plot()
print("Preetha S, 212222230110")
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("Preetha S, 212222230110")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('mnist.jpg')
type(img)
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print('PREETHA S')
print(x_single_prediction)

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-09-16 104604](https://github.com/user-attachments/assets/c79cabd0-7ce6-4155-aff7-2c348237bfce)

![Screenshot 2024-09-16 104621](https://github.com/user-attachments/assets/536f443c-bb5a-4044-bf9a-94d2961d61e0)

![Screenshot 2024-09-16 104630](https://github.com/user-attachments/assets/85fb0e00-7031-45ff-949a-4a0a7bc2ea01)


### Classification Report

![Screenshot 2024-09-16 104642](https://github.com/user-attachments/assets/5cac1588-87fa-481b-b033-1f9c936c8458)


### New Sample Data Prediction

![Screenshot 2024-09-16 104655](https://github.com/user-attachments/assets/de84283c-fd59-4fdc-ac52-663b2cc22315)


## RESULT

A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
