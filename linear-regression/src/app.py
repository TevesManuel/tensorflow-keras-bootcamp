import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np

import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input, Activation
# from tensorflow.keras.datasets import boston_housing
# from tensorflow.keras import layers

import matplotlib.pyplot as plt

print("[i] libs are loaded.")

SEED_VALUE = 42

# Fix seed to make training deterministic.
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Load the dataset.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# "Average Number of Rooms" label is the 5 row
# Extract "Average Number of Rooms" of the X_train rows 
X_train_1d = X_train[:, 5]
print(X_train_1d.shape)
X_test_1d = X_test[:, 5]


plt.figure(figsize=(15, 5))
plt.xlabel("Average Number of Rooms")
plt.ylabel("Median Price [$K]")
plt.grid("on")
plt.scatter(X_train_1d[:], y_train, color="green", alpha=0.5)
plt.title("Train dataset")
plt.show()