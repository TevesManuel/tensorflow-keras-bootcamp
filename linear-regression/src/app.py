import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

print("[i] libs are loaded.")

SEED_VALUE = 42

# Why i set a static seed
# Reproducibility:
#   Allows others to reproduce the same results when
#   running the same code, which is critical in research and development.
# Debugging:
#   Makes it easier to detect and fix errors because results do not change between runs.
# Consistency in model training:
#   In machine learning, setting a seed ensures that the model is trained
#   with the same training data set and the same initial weights each time,
#   providing consistency in results and fair comparisons.

np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Load the dataset.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# "Average Number of Rooms" label is the 5 row
# Extract "Average Number of Rooms" of the X_train rows 
X_train_1d = X_train[:, 5]
X_test_1d = X_test[:, 5]

# Print the shape of the X_train and X_train_1d dataset for view the difference between rows
print(X_train.shape)
print(X_train_1d.shape)


# Graph the training data set
plt.figure(figsize=(15, 5))
plt.xlabel("Average Number of Rooms")
plt.ylabel("Median Price [$K]")
plt.grid("on")
plt.scatter(X_train_1d[:], y_train, color="green", alpha=0.5)
plt.title("Train dataset")
plt.show()

# Create the Sequential type model
model = tf.keras.models.Sequential()

# Add a dense layer with a single neuron to the model
model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

# Print the architecture of the model
print("[i] Model architecture:\n")
print(str(model.summary()) + "\n\n")

# Compilate the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005), loss="mse")

# Fit the model & return the history of the train
history = model.fit(
    X_train_1d, 
    y_train, 
    batch_size=16, 
    epochs=101, 
    validation_split=0.3,
)

# Graph the loss of the train

plt.figure(figsize=(20,5))
plt.plot(history.history['loss'], 'g', label='Training Loss')
plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
plt.xlim([0, 100])
plt.ylim([0, 300])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Makes some predictions & print the out
x = [3, 4, 5, 6, 7]
y_pred = model.predict(x)
for idx in range(len(x)):
    print(f"Predicted price of a home with {x[idx]} rooms: ${int(y_pred[idx] * 10) / 10}K")

# Generate feature data that spans the range of interest for the independent variable.
x = np.linspace(3, 9, 10)

# Use the model to predict the dependent variable.
y = model.predict(x)

# Transform a tuple 
modelPredictions = (x, y)

# Function to make a graph
def pltCompareDatasetWithModel(Dataset_x, Dataset_y, modelPredicts):
    plt.figure(figsize=(15,5))
    plt.scatter(Dataset_x, Dataset_y, label='Ground Truth', color='green', alpha=0.5)
    plt.plot(modelPredicts[0], modelPredicts[1], color='k', label='Model Predictions')
    plt.xlim([3,9])
    plt.ylim([0,60])
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Price [$K]')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot the difference between predictions and true value

pltCompareDatasetWithModel(X_train_1d, y_train, modelPredictions)
pltCompareDatasetWithModel(X_test_1d, y_test, modelPredictions)