import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
import pickle


def main():
    x_train = pickle.load(open("cats_and_dogs_challenge/X.pickle", "rb"))
    y_train = pickle.load(open("cats_and_dogs_challenge/Y.pickle", "rb"))
    x_train = x_train / 255

    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(60, 60, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('sigmoid'))
    model.add(Dense(84))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
    model.save("C:/Users/benme/OneDrive/Desktop/Code/cats_and_dogs_challenge")
main()