import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from dataclasses import dataclass

if __name__ == '__main__':
    #Initialice SEED so that the results do not vary
    SEED_VALUE = 42

    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    tf.random.set_seed(SEED_VALUE)

    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = cifar10.load_data()

    #View some figures
    # plt.figure(figsize=(18, 8))

    # num_rows = 4
    # num_cols = 8

    # for i in range(num_rows * num_cols):
    #     ax = plt.subplot(num_rows, num_cols, i + 1)
    #     plt.imshow(X_TRAIN[i, :, :])
    #     plt.axis("off")

    # plt.show()

    #Preprocess data
    # Normalize images to the range [0, 1].
    X_TRAIN = X_TRAIN.astype("float32") / 255
    X_TEST  = X_TEST.astype("float32")  / 255

    # Change the labels from integer to categorical data.
    print('Original (integer) label for the first training sample: ', Y_TRAIN[0])

    # Convert labels to one-hot encoding.
    Y_TRAIN = to_categorical(Y_TRAIN)
    Y_TEST  = to_categorical(Y_TEST)

    print('After conversion to categorical one-hot encoded labels: ', Y_TRAIN[0])

    @dataclass(frozen=True)
    class DatasetConfig:
        NUM_CLASSES  : int = 10
        IMG_HEIGHT   : int = 32
        IMG_WIDTH    : int = 32
        NUM_CHANNELS : int = 3
        
    @dataclass(frozen=True)
    class TrainingConfig:
        EPOCHS       : int = 31
        BATCH_SIZE   : int = 256
        LEARNING_RATE: float = 0.001
    

    input_shape = (32, 32, 3)

    model = Sequential()

    #------------------------------------
    # Conv Block 1: 32 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #------------------------------------
    # Conv Block 2: 64 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #------------------------------------
    # Conv Block 3: 64 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    #------------------------------------
    # Flatten the convolutional features.
    #------------------------------------
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    # To view the structure of the model
    # model.summary()

    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_TRAIN,
        Y_TRAIN,
        batch_size=TrainingConfig.BATCH_SIZE, 
        epochs=TrainingConfig.EPOCHS, 
        verbose=1, 
        validation_split=.3,
    )

    def plot_results(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
        fig, ax = plt.subplots(figsize=(15, 4))

        if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
            metrics = [metrics,]
            metric_name = [metric_name,]

        for idx, metric in enumerate(metrics):
            ax.plot(metric, color=color[idx])

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlim([0, TrainingConfig.EPOCHS - 1])
        plt.ylim(ylim)
        # Tailor x-axis tick marks
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        plt.grid(True)
        plt.legend(metric_name)
        plt.show()
        plt.close()

    train_loss = history.history["loss"]
    train_acc  = history.history["accuracy"]
    valid_loss = history.history["val_loss"]
    valid_acc  = history.history["val_accuracy"]

    model.save("./Model.keras")
    model.save("./Model.h5")

    plot_results(
        [train_loss, valid_loss],
        ylabel="Loss",
        ylim=[0.0, 5.0],
        metric_name=["Training Loss", "Validation Loss"],
        color=["g", "b"],
    )

    plot_results(
        [train_acc, valid_acc],
        ylabel="Accuracy",
        ylim=[0.0, 1.0],
        metric_name=["Training Accuracy", "Validation Accuracy"],
        color=["g", "b"],
    )