import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
import pickle
import os
import cv2

def main():
    dir = "C:/Users/benme/OneDrive/Desktop/Code/cats_and_dogs_challenge/TestImages"
    data = []
    for img in os.listdir(dir):
        img_array = cv2.imread(os.path.join(dir, img), cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img_array, (60, 60))
        data.append([resized_img, 0]) # The test data is 3 pictures of my dogs.
    x_test = []
    y_test = []

    for features, label in data:
        x_test.append(features)
        y_test.append(label)

    x_test = np.array(x_test).reshape(-1, 60, 60, 1)
    y_test = np.array(y_test)
    x_test = x_test / 255

    model = keras.models.load_model("C:/Users/benme/OneDrive/Desktop/Code/cats_and_dogs_challenge")
    res = model.predict(x_test)
    print(res)
main()