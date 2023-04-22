import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
import random
import pickle

def main():
    dir = "C:/Users/benme/OneDrive/Desktop/Code/cats_and_dogs_challenge/PetImages"
    categories = ["Dog", "Cat"]

    data = []

    for category in categories:
        path = os.path.join(dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_img = cv2.resize(img_array, (60, 60))
                data.append([resized_img, class_num])
            except Exception as e:
                pass
    
    random.shuffle(data)
    x_train = []
    y_train = []
    for features, label in data:
        x_train.append(features)
        y_train.append(label)

    x_train = np.array(x_train).reshape(-1, 60, 60, 1)
    y_train = np.array(y_train)

    pickle_out = open("cats_and_dogs_challenge/X.pickle", "wb")
    pickle.dump(x_train, pickle_out)
    pickle_out.close()

    pickle_out = open("cats_and_dogs_challenge/Y.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()
main()
